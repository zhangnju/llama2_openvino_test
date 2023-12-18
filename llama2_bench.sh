#!/bin/bash
promptfile="/home/test.txt"
model_path="/home/llama2.openvino/ir_model_fp16"
instance=4
PRECISION=fp16 

rm llama2_${PRECISION}_instance_*.log
rm llama2_summary_${PRECISION}_instance${instance}.log
if [[ $instance == 2 ]]; then
    echo "running in 2 instances"
    numactl --physcpubind=0-95,192-287 python3 llama2_inference.py -m ${model_path} -p ${promptfile} -d "CPU" > llama2_${PRECISION}_instance_0.log 2>&1 &
    numactl --physcpubind=96-191,288-383 python3 llama2_inference.py -m ${model_path} -p ${promptfile} -d "CPU" > llama2_${PRECISION}_instance_1.log 2>&1
elif [[ $instance == 4  ]]; then
    echo "running in 4 instances"
    numactl --physcpubind=0-95 python3 llama2_inference.py -m ${model_path} -p ${promptfile} -d "CPU" > llama2_${PRECISION}_instance_0.log 2>&1 &
    numactl --physcpubind=192-287 python3 llama2_inference.py -m ${model_path} -p ${promptfile} -d "CPU" > llama2_${PRECISION}_instance_1.log 2>&1 &
    numactl --physcpubind=96-191 python3 llama2_inference.py -m ${model_path} -p ${promptfile} -d "CPU" > llama2_${PRECISION}_instance_2.log 2>&1 &
    numactl --physcpubind=288-383 python3 llama2_inference.py -m ${model_path} -p ${promptfile} -d "CPU" > llama2_${PRECISION}_instance_3.log 2>&1
else
    echo "running in 1 instances"
    numactl --physcpubind=0-383 python3 llama2_inference.py -m ${model_path} -p ${promptfile} -d "CPU" > llama2_${PRECISION}_instance_0.log 2>&1
fi
    
wait
Tokens=$(grep 'token:'  llama2_${PRECISION}_instance_* |sed -e 's/.*token//;s/[^0-9.]//g' |awk '
    BEGIN {
        sum = 0;
        i = 0;
        }
        {
            sum = sum + $1;
            i++;
        }
    END   {
        #sum = sum / i;
        printf("%.3f", sum);
    }')

Seconds=$(grep 'seconds:'  llama2_${PRECISION}_instance_* |sed -e 's/.*seconds//;s/[^0-9.]//g' |awk '
    BEGIN {
        max = 0;
        }
        {
            if( $1 > max) max = $1;
        }
    END   {
        printf("%.3f", max);
    }')
latency=$(grep 'latency:'  llama2_${PRECISION}_instance_* |sed -e 's/.*latency//;s/[^0-9.]//g' |awk '
    BEGIN {
        sum = 0;
        i = 0;
        }
        {
            sum = sum + $1;
            i++;
        }
    END   {
        sum = sum / i;
        printf("%.3f", sum);
    }')
echo "infernece max seconds: ${Seconds}"
echo "generate total tokens: ${Tokens}"
throughput=`echo "scale=3; $Tokens/$Seconds" | bc`
echo "Llama2-7B precison:${PRECISION} throughput (Token/Second):" "${throughput}" "avg latency (ms/token):" "${latency}" "ms" | tee -a llama2_summary_${PRECISION}_instance${instance}.log

