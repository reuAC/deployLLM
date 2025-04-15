# 章节 一

## 前提条件

- 操作系统：Ubuntu 24.04

## 文章约定

- 命令以行为单位
- 在输入完成命令后需要按下 `Enter` 以完成使用

## 1.1 vLLM安装

在进入系统后，建议使用如下指令进行系统软件包更新：

```shell
sudo apt update
sudo apt upgrade -y
```



#### 1.1.1 安装显卡驱动

运行 vLLM 需要安装显卡驱动，建议前往 https://www.nvidia.cn/drivers/lookup/ ，在其中手动选择显卡型号，平台选择 Linux 64-bit Ubuntu 24.04

示例中，采用了 Tesla T4 

![](第一章\1.png)

按下查找后，会出现如下画面，点击查看

![](第一章\2.png)

随后会有该画面，右键下载按钮，点击**复制链接地址**

![](第一章\3.png)

回到控制台，在地址前加上 `wget` 进行下载。

```shell
wget <链接>
```

示例（使用了 Tesla T4 驱动的地址）

```shell
wget https://cn.download.nvidia.com/tesla/570.124.06/nvidia-driver-local-repo-ubuntu2404-570.124.06_1.0-1_amd64.deb
```



下载完成后，使用如下指令进行安装

```shell
sudo dpkg -i <文件名>
```

示例（使用了 Tesla T4 驱动文件）

```shell
sudo dpkg -i nvidia-driver-local-repo-ubuntu2404-570.86.15_1.0-1_amd64.deb
```

安装完成后，会有一段需要输入的代码，位置在 `To install the key, run this command:` 的下一行（以sudo开头的指令）
示例中的相关内容如下

```shell
To install the key, run this command:
sudo cp /var/nvidia-driver-local-repo-ubuntu2404-570.86.15/nvidia-driver-local-41F54E74-keyring.gpg /usr/share/keyrings/
```

此时需要执行如下命令

```shell
sudo cp /var/nvidia-driver-local-repo-ubuntu2404-570.86.15/nvidia-driver-local-41F54E74-keyring.gpg /usr/share/keyrings/
```

完成后，更新软件源

```shell
sudo apt update
```

安装驱动

```shell
sudo apt install nvidia-driver-570
```

安装完成后需要重启

```shell
sudo reboot
```

重启完成后，可以使用如下命令查看显卡状态与判断安装是否成功

```shell
nvidia-smi
```

若输出形如表格的内容即为成功。

示例

![](第一章\4.png)



- Tue Mar 4 10:17:05 2025: 显示当前日期和时间。

- NVIDIA-SMI 570.86.15: NVIDIA 系统管理界面的版本号。

- Driver Version: 570.86.15: 安装的 NVIDIA 驱动程序版本。

- CUDA Version: 12.8: 安装的 CUDA 工具包版本。



**GPU 状态部分（表格形式）:**

以等号为下框线的单元格行即为表头，它给出了其下方表格对应位置的含义。

每一行代表一个 GPU（图形处理器）。

- **GPU**: GPU 编号（从0开始，这里有 0, 1, 2, 3，共四个GPU）。
- **Name**: GPU 型号（这里都是 Tesla T4）。
- **Fan**: 风扇转速（这里是 N/A，表示不可用或不支持风扇转速监控）。
- **Temp**: GPU 温度（单位：摄氏度）。
- **Perf**: 性能状态（P0 表示最高性能模式, 数字越小性能越高）。
- **Persistence-M**: 持久模式（Off，表示关闭）。持久模式可以使驱动程序在没有应用程序使用 GPU 时保持加载状态，从而减少后续 GPU 应用启动的延迟。
- **Pwr:Usage/Cap**: 功耗信息。 例如 "28W / 70W" 表示当前功耗 28 瓦，最大功耗限制为 70 瓦。
- **Bus-Id**: GPU 的 PCI 总线 ID（用于标识 GPU 在系统中的位置）。
- **Disp.A**: 显示活动状态（Off，表示关闭）。指示是否有显示器连接到该 GPU。
- **Memory-Usage**: 显存使用情况。例如 "14451MiB / 15360MiB" 表示已使用 14451 MiB 显存，总共有 15360 MiB 显存。
- **GPU-Util**: GPU 利用率（0%）。表示 GPU 的计算核心在过去一段时间内的繁忙程度（百分比）。
- **Volatile Uncorr. ECC**: ECC 错误计数。ECC 是一种内存纠错技术。这里显示的是无法纠正的 ECC 错误数量（通常为0表示没有错误）。
- **Compute M.**: 计算模式（Default，表示默认模式）。
- **MIG M.**: 多实例 GPU 模式（示例中的 Tesla T4 上不支持）。



**进程部分（Processes）:**

显示当前正在使用 GPU 的进程。

- **GPU**: 进程正在使用的 GPU 编号。
- **GI ID**: GPU实例ID。
- **CI ID**: 计算实例ID。
- **PID**: 进程 ID（操作系统分配给每个进程的唯一编号）。
- **Type**: 进程类型（"C" 表示计算进程，即主要进行计算任务的进程）。
- **Process name**: 进程名称（这里都是 /LLM/main/bin/python3，此处是使用了vLLM运行大模型）。
- **GPU Memory Usage**: 该进程使用的显存量。



推荐使用 `nvtop` 监控显卡占用情况，可以使用如下指令安装

```shell
sudo apt install nvtop
```

使用如下指令以使用（依照下方提示，按下 F10 即可退出）

```shell
nvtop
```



#### 1.1.2 安装Python等依赖项

因为Ubuntu 24.04中自带Python3.12，故如下步骤是可以忽略的。

*（可忽略）*安装 Python3.12

```shell
sudo apt install python3
```



安装 Python3-pip 与 Python3.12-venv 虚拟环境。

```shell
sudo apt install python3-pip python3.12-venv
```



在安装过程中，有可能会有如下提示，用于向用户确认是否继续进行安装，此时需要输入 `y` 并回车来确认继续进行安装

![](第一章\5.png)



在合适位置创建目录，并切换到该目录。

```shell
sudo mkdir -p <位置>
cd <位置>
```

例子：

```shell
mkdir -p ~/llm
cd ~/llm
```



#### 1.1.2-R-1*（推荐）*配置Pypi镜像软件源

配置加速镜像能够提高在国内的下载速度。

此处使用的镜像来自[中国科学技术大学Mirror](https://telecom.mirrors.ustc.edu.cn/)
其中Pypi镜像软件源说明文档来自：https://telecom.mirrors.ustc.edu.cn/help/pypi.html

临时使用该镜像软件源

```shell
pip install -i https://mirrors.ustc.edu.cn/pypi/simple package
```

设为默认软件源，默认使用该软件源

```shell
pip install -i https://mirrors.ustc.edu.cn/pypi/simple pip -U
pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/simple
```





#### 1.1.3 Python虚拟环境的使用

在该位置创建虚拟环境并进入。

```shell
python3 -m venv <虚拟环境名称>
source <虚拟环境名称>/bin/activate
```

例子：

```shell
python3 -m venv main
source main/bin/activate
```



#### 1.1.4 安装vLLM

使用指令安装vLLM

```shell
pip install vllm
```



验证安装是否成功

```shell
python3 -c "import vllm; print(vllm.__version__)"
```

若提示形如 `0.7.3` 的版本号，而没有其他内容，即为安装成功。



## 1.2 vLLM使用

#### 1.2-R-1 vLLM运行注意事项

- 当单张显卡显存不足时，建议使用 `tensor-parallel-size <数量>` 参数，使得多卡协同运行。
- 建议使用基础模型的 tokenizer 而不是 GGUF 模型的 tokenizer。因为从 GGUF 转换 tokenizer 既耗时又不稳定，尤其是对于一些词汇量较大的模型。

#### 1.2.1 下载模型

推荐使用 `modelscope` 提供的方式进行模型的下载。

安装 ModelScope

```shell
pip install modelscope
```



推荐前往 https://www.modelscope.cn/ 平台搜索所需的模型，同时注意其版本。
此处采用 `DeepSeek-R1-Distill-Qwen-1.5B` 作为示例。



下载模型

```shell
modelscope download --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```



下载完成后，寻找控制台中的`Downloading Model to directory:` 其后紧跟着模型存放位置的软链接
例如：`Downloading Model to directory: /root/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`

此时使用指令将软链接复制到现在所在的位置

```shell
cp -r /root/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B .
```



#### 1.2.2 基于代码使用 vLLM

基于 Python 代码使用的示例

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# 从预训练模型 "DeepSeek-R1-Distill-Qwen-1.5B" 加载分词器。
# 在这里 "DeepSeek-R1-Distill-Qwen-1.5B" 指的是文件位置，此处使用了相对位置。
tokenizer = AutoTokenizer.from_pretrained("DeepSeek-R1-Distill-Qwen-1.5B")

# 设置采样参数。
# temperature：控制生成文本的随机性，值越高，随机性越大。
# top_p：控制生成文本的多样性，值越高，多样性越大。
# repetition_penalty：控制生成文本中重复内容的程度，值越高，重复内容越少。
# max_tokens：生成文本的最大长度。
sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)

# 初始化 LLM 对象，加载模型 "DeepSeek-R1-Distill-Qwen-1.5B"。
# 在这里 "DeepSeek-R1-Distill-Qwen-1.5B" 指的是文件位置，此处使用了相对位置。
# model 参数可以是模型名称或路径，支持 GPTQ 或 AWQ 模型。
# 参数可参考文档：https://vllm.hyper.ai/docs/serving/openai-compatible-server/
# 在代码中，参数应当添加在下一行之中，示例：
# llm = LLM(model="DeepSeek-R1-Distill-Qwen-32B",dtype="half")
# 可以以逗号为分隔使多个参数同时生效
llm = LLM(model="DeepSeek-R1-Distill-Qwen-1.5B")

# 定义提示文本。
prompt = "你好呀."

# 构建对话历史。
messages = [
    {"role": "system", "content": "You are a helpful assistant."},  # 系统角色设定：你是一个有帮助的助手。
    {"role": "user", "content": prompt}  # 用户角色：输入提示文本。
]

# 使用分词器的 apply_chat_template 方法将对话历史转换为模型输入的文本格式。
# tokenize=False：不进行分词，因为我们只需要文本格式。
# add_generation_prompt=True：添加生成提示，指示模型开始生成。
#  注意:  不同的模型可能需要不同的 chat template 格式, 需要查阅模型文档确定.  
#        如果模型不兼容,  可能需要手动构建 prompt.
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# 使用 LLM 对象的 generate 方法生成文本。
# 输入为包含格式化后提示文本的列表和采样参数。
outputs = llm.generate([text], sampling_params)

# 遍历输出结果。
for output in outputs:
    prompt = output.prompt  # 获取输入的提示文本。
    generated_text = output.outputs[0].text  # 获取生成的文本。
    print(f"提示文本: {prompt!r}, 生成的文本: {generated_text!r}")  # 打印提示文本和生成的文本。
```

在合适位置，使用 vi 或 vim 指令创建代码文件，并将如上代码复制到其中
注意：建议参考注释，修改模型文件位置、启动参数等。



创建代码文件

```
vim <文件名>.py
```

示例

```shell
vim run_deepseek_1.5B.py
```

随后按下 `i` 键进入输入模式，将代码复制到其中，可选的修改内容，随后按下 `ESC` 键，输入 `:wq` 保存并退出。



运行代码文件

```shell
python3 <文件名>.py
```

示例

```shell
python3 run_deepseek_1.5B.py
```

随后即会在控制台中输出处理结果。



#### 1.2.3 vLLM 建立兼容OpenAI API的服务

使用如下指令以建立兼容OpenAI API的服务，基于vLLM。

```shell
vllm serve DeepSeek-R1-Distill-Qwen-1.5B
```





可以通过添加参数实现一定需求
下文仅展示了三个参数，如需设定更多项，可参考文档：https://vllm.hyper.ai/docs/serving/openai-compatible-server/

设置密钥

```shell
vllm serve DeepSeek-R1-Distill-Qwen-1.5B --api-key <密钥>
```



配置模型权重和激活的数据类型，示例为自动

```shell
vllm serve DeepSeek-R1-Distill-Qwen-1.5B --dtype auto
```

部分显卡不支持bfloat16（如Tesla T4），会导致服务启动错误，并会有提示将类型设置为half，即为如下示例

```shell
vllm serve DeepSeek-R1-Distill-Qwen-1.5B --dtype half
```



张量并行副本的数量（可多卡协同运行大模型）

```shell
vllm serve DeepSeek-R1-Distill-Qwen-1.5B --tensor-parallel-size <数量>
```



可以以空格为分隔使它们同时生效

```shell
vllm serve DeepSeek-R1-Distill-Qwen-1.5B --dtype auto --api-key token-abc123 --tensor-parallel-size 4
```



#### 1.2.4 运行GGUF量化模型及其分词器的注意事项

参考文档：https://vllm.hyper.ai/docs/quantization/gguf



目前，vLLM 仅支持加载单文件 GGUF 模型。如果您有多文件的 GGUF 模型，可以使用 [gguf-split](https://github.com/ggerganov/llama.cpp/pull/6135) 工具将其合并为一个单文件模型。

```
vllm serve <GGUF文件位置>
```

示例

```shell
vllm serve DeepSeek-R1-Distill-Qwen-32B-Q8_0.gguf 
```



建议采用基础模型的 tokenizer，只需要下载基础模型的 `tokenizer_config.json` 与 `tokenizer.json`，并将其放置在同一目录下。

```shell
vllm serve DeepSeek-R1-Distill-Qwen-32B-Q8_0.gguf --tokenizer <tokenizer目录位置>
```

示例命令中假设该目录名为tokenizer_base。

```shell
vllm serve DeepSeek-R1-Distill-Qwen-32B-Q8_0.gguf --tokenizer tokenizer_base
```



#### 1.2-R-2 修正基于vLLM部署Deepseek时，思维链在部分平台无法正常渲染

原因分析：因为在 `tokenizer_config.json` 中，强制在用户输出之后，加入了<think>标签作为输入令模型输出，故模型不会再次输出<think>标签，仅会有</think>，故得到的模型输出无法渲染正确的思维链样式。

官方原文：此外，我们观察到 DeepSeek-R1 系列模型在回应某些查询时，倾向于绕过思考模式（即省略输出“<think>\n\n</think>”），这可能会对模型的性能产生不利影响。为确保模型进行充分的推理，我们建议强制模型在每次输出的开头都以“<think>\n”启动其回应。

vLLM已给出了专门的解决方法：https://docs.vllm.ai/en/latest/features/reasoning_outputs.html#quickstart
在启动参数中添加`--enable-reasoning --reasoning-parser deepseek_r1`即可



##### 不再适用的解决方法

使用 `vi` 或 `vim` 打开模型目录内的 `tokenizer_config.json` 。

```shell
vim tokenizer_config.json
```

找到并将光标移动到 ` "chat_template" ` 所在的行（一般是最后一行），双击 `d` 以删除改行，随后按下 `i` 进入输入模式，将如下内容粘贴进去。

```json
"chat_template": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='', is_first_sp=true) %}{%- for message in messages %}{%- if message['role'] == 'system' %}{%- if ns.is_first_sp %}{% set ns.system_prompt = ns.system_prompt + message['content'] %}{% set ns.is_first_sp = false %}{%- else %}{% set ns.system_prompt = ns.system_prompt + '\n\n' + message['content'] %}{%- endif %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{{'<｜Assistant｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜>'}}{% endif %}"
```

随后按下 `ESC` 键，输入 `:wq` 保存并退出。

重启 vLLM 服务即可。



#### 1.2.5 后台运行 vLLM

