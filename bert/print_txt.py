import os
import subprocess

# 定义要使用的模型名称列表
# models = ["bert","bert_CNN","bert_DPCNN", "bert_RCNN","bert_RNN", "ERNIE","ERNIE3-base-zh_RCNN","ERNIE3nano_RCNN", "ERNIE3xbase_RCNN", "ERNIE_RCNN",  "ERNIE_RCNN_improved", "ERNIE_RCNN_roberta_chinese", "ERNIE_RCNN_roberta_chinese_improved" ,"ERNIE_RCNN_roberta_chinese_large", "ERNIE_RCNN_wwm_chinese", "ERNIE_RNN"]
models = ["ERNIE", "ERNIE_RCNN",  "ERNIE_RCNN_improved", "ERNIE_RNN"]
# 确保结果保存的文件夹存在
result_dir = "result-train"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 依次遍历每个模型名称并运行相应的命令
for model in models:
    # 构建命令
    command = f"python run.py --model \"{model}\""
    print(f"Running: {command}")

    # 打开对应的结果文件，保存运行结果
    result_file_path = os.path.join(result_dir, f"{model}.txt")
    with open(result_file_path, "w") as result_file:
        try:
            # 使用 subprocess 运行命令，并将标准输出保存到对应的txt文件
            process = subprocess.run(command, shell=True, stdout=result_file, stderr=subprocess.STDOUT, text=True)
            print(f"Finished running model: {model}, result saved to {result_file_path}")
        except Exception as e:
            # 记录异常信息到文件
            result_file.write(f"Error occurred while running model {model}: {str(e)}")
            print(f"Error occurred while running model {model}: {str(e)}")
