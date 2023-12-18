from transformers import LlamaTokenizer
from openvino.runtime import Core, Tensor
from pathlib import Path
import numpy as np
import argparse
import time


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    summation = e_x.sum(axis=-1, keepdims=True)
    return e_x / summation


def process_logits(cur_length, scores, eos_token_id, min_length=0):
    if cur_length < min_length:
        scores[:, eos_token_id] = -float("inf")
    return scores


def get_top_k_logits(scores, top_k):
    filter_value = -float("inf")
    top_k = min(max(top_k, 1), scores.shape[-1])
    top_k_scores = -np.sort(-scores)[:, :top_k]
    indices_to_remove = scores < np.min(top_k_scores)
    filtred_scores = np.ma.array(scores,
                                 mask=indices_to_remove,
                                 fill_value=filter_value).filled()
    return filtred_scores


def generate_sequence(sampling, input_ids, attention_mask, eos_token_id,
                      max_sequence_length):
    past_key_values = None
    prompt_len = len(input_ids[0])
    count = 0
    while True:
        inputs = {}
        if past_key_values is not None:
            inputs = dict(zip(key_value_input_names, past_key_values))
            inputs["input_ids"] = input_ids[:, -1:]
            cur_input_len = 1
        else:
            inputs["input_ids"] = input_ids
            shape_input_ids = input_ids.shape
            num_attention_heads = 1
            for input_name in key_value_input_names:
                model_inputs = model.input(input_name)
                shape = model_inputs.get_partial_shape()
                shape[0] = shape_input_ids[0] * num_attention_heads
                if shape[2].is_dynamic:
                    shape[2] = 0
                if shape[1].is_dynamic:
                    shape[1] = 0
                inputs[input_name] = Tensor(model_inputs.get_element_type(),
                                            shape.get_shape())
        cur_input_len = len(inputs["input_ids"][0])
        if "attention_mask" in input_names and attention_mask is not None:
            inputs["attention_mask"] = attention_mask
        request.start_async(inputs, shared_memory=True)
        request.wait()
        count += 1
        logits = request.get_tensor("logits").data
        past_key_values = tuple(
            request.get_tensor(key).data for key in key_value_output_names)
        #print(past_key_values[0].shape)
        next_token_logits = logits[:, cur_input_len - 1, :]
        # pre-process distribution
        next_token_scores = process_logits(len(input_ids[0]),
                                           next_token_logits, eos_token_id)
        top_k = 20
        next_token_scores = get_top_k_logits(next_token_scores, top_k)
        # get next token id
        if sampling:
            probs = softmax(next_token_scores)
            next_tokens = np.random.choice(probs.shape[-1],
                                           1,
                                           p=probs[0],
                                           replace=True)
        else:
            next_tokens = np.argmax(next_token_scores, axis=-1)
        # break the loop if max length or end of text token is reached
        if (len(input_ids[0]) - prompt_len
            ) == max_sequence_length or next_tokens == eos_token_id:
            break
        else:
            input_ids = np.concatenate((input_ids, [next_tokens]), axis=-1)
            attention_mask = np.concatenate(
                (attention_mask, [[1] * len(next_tokens)]), axis=-1)
    return input_ids, count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h',
                        '--help',
                        action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-m',
                        '--model_path',
                        required=True,
                        type=str,
                        help='Required. path of IR model and tokenizer')
    parser.add_argument('-p',
                        '--prompt_file',
                        required=True,
                        type=str,
                        help='Required. prompt sentence file')
    parser.add_argument('-l',
                        '--max_sequence_length',
                        default=128,
                        required=False,
                        type=int,
                        help='maximun lengh of output')
    parser.add_argument('-d',
                        '--device',
                        default='CPU',
                        required=False,
                        type=str,
                        help='device for inference')
    parser.add_argument('-s',
                        '--sampling',
                        default=True,
                        required=False,
                        type=bool,
                        help='sampling or not')
    args = parser.parse_args()

    
    core = Core()
    ir_model_path = Path(args.model_path)
    ir_model = ir_model_path / "openvino_model.xml"
    
    
    print(" --- reading model --- ")
    # read the model and corresponding weights from file
    model = core.read_model(ir_model)
    
    input_names = {
        key.get_any_name(): idx
        for idx, key in enumerate(model.inputs)
    }
    
    output_names = {
        key.get_any_name(): idx
        for idx, key in enumerate(model.outputs)
    }
    key_value_input_names = [key for key in input_names if "key_values" in key]
    key_value_output_names = [key for key in output_names if "present" in key]
    
    print(" --- model compiling --- ")
    # compile the model for CPU devices
    request = core.compile_model(
        model=model, device_name=args.device).create_infer_request()

    print(" --- load tokenizer --- ")
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
    inference_count=0
    total_token =0
    total_second =0
    latency =0
    #inference_start = time.perf_counter()
    with open(args.prompt_file) as f:
        for prompt in f:
            inference_count +=1
            inputs = tokenizer(prompt, return_tensors="np")
            #print(inputs["input_ids"].shape)
            #print(inputs["attention_mask"].shape)

            #print(" --- start generating --- ")
            start = time.perf_counter()
            output_ids, num_tokens = generate_sequence(
                args.sampling,
                inputs["input_ids"],
                inputs["attention_mask"],
                eos_token_id=tokenizer.eos_token_id,
                max_sequence_length=args.max_sequence_length,
            )
            end = time.perf_counter()
            total_token += num_tokens
            seconds = end-start 
            latency += seconds*1000.0 /num_tokens
            total_second +=seconds 
            #print( f"Generated {num_tokens} tokens in {end - start:.3f} s on {args.device}")
            """
            # Convert IDs to words and make the sentence from it
    
            print(" --- text decoding --- ")
            output_text = tokenizer.batch_decode(output_ids,
                                         skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)[0]
    
            print(f"Response: {output_text}")
            """
    
    #inference_stop = time.perf_counter()
    #inference_seconds = inference_stop - inference_start
    avg_latency = latency/inference_count 
    print(f"avg latency:{avg_latency:.3f} ms")
    print(f"infernece seconds:{total_second:.3f}")
    print(f"generated total token:{total_token}")