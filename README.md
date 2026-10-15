# 大模型微调实验工作区

用于存放大模型微调的数据、配置、训练脚本与实验产物。

## TNEWS 中文新闻分类实验

模型为 `Qwen/Qwen2.5-1.5B-Instruct`，任务是根据中文标题生成
`{"category":"体育"}` 这样的 JSON。使用 BF16、单卡 LoRA，冻结原模型，
仅训练注意力和 MLP 线性层的适配器（rank 8）。损失只计算助手答案及结束标记。
下方 0.5B + Capybara 是较早准备的通用对话示例。

### 多轮训练与曲线记录（2026-10-01）

已完成 3 轮、375 步训练，Trainer 记录耗时 516.5 秒（包含轮末验证和保存）。
数据文件哈希与首次单轮实验一致，三个检查点均已保存，曲线包含 75 个训练日志点和 4 个验证点。

| 轮次 | 验证 loss | 验证准确率 | 验证格式及类别合法率 |
| --- | --- | --- | --- |
| 训练前 | 0.5336 | 37.5% | 84.0% |
| 第 1 轮 | 0.2148 | 56.5% | 96.0% |
| 第 2 轮 | 0.1952 | 56.5% | 99.5% |
| 第 3 轮 | 0.2084 | 56.5% | 99.5% |

三轮验证准确率相同，因此按预设规则选中第 1 轮（`checkpoint-125`）。
最终 `adapter/` 的权重文件校验值与该检查点完全一致，并已重新加载评估。
其测试准确率为 58.0%（174/300），格式及类别合法率 97.33%，macro-F1 为 0.5373。
这些测试结果对应选出的第 1 轮权重，并非第 3 轮权重。
`metrics.json` 的 `validation_after` 也对应所选权重；其中 `epoch: 3` 是 Trainer 已完成的训练轮数，
实际模型轮次以 `selection.best_epoch` 为准。

重复训练降低了训练 loss，但第 2、3 轮没有提高验证准确率，第 3 轮验证 loss 回升，
出现过拟合迹象。本次第 1 轮的学习率计划与先前单轮实验不同，不能将两次测试分数的差异归因于增加轮数。

[查看 loss 曲线](outputs/tnews_qwen15b_lora_3epochs_20261001/loss_curve.png) ·
[查看全部曲线](outputs/tnews_qwen15b_lora_3epochs_20261001/training_curves.png) ·
[PDF](outputs/tnews_qwen15b_lora_3epochs_20261001/training_curves.pdf) ·
[训练 CSV](outputs/tnews_qwen15b_lora_3epochs_20261001/training_history.csv) ·
[验证 CSV](outputs/tnews_qwen15b_lora_3epochs_20261001/validation_history.csv)

新增实验 `tnews_qwen15b_lora_3epochs_20261001` 从原始模型重新初始化 LoRA，
总共训练 3 个 epoch（375 次参数更新），沿用之前相同的 2,000/200/300 条数据划分。
每 5 步记录训练 loss、学习率、梯度范数，每轮计算验证 loss、严格分类准确率、
macro-F1 和格式合规率，并保存一个包含优化器状态的检查点。

模型按验证准确率最高选择，相同准确率时保留较早检查点；测试集只在训练前和选模后评估。
`adapter/` 保存的是验证集选中的模型，可能并非第 3 轮模型。
余弦学习率计划按新的 375 步计算，因此本次第 1 轮的学习率轨迹与之前单轮实验不同。

在 GPU 作业中运行（新实验必须换一个未使用的名字）：

```bash
bash scripts/run_tnews.sh tnews_qwen15b_lora_3epochs_20261001 3
```

新增输出文件均位于 `outputs/实验名/`：

- `loss_curve.png`：训练和验证 loss 曲线；训练点是最近 5 步的日志均值，验证点覆盖整个验证集。
- `training_curves.png`、`training_curves.pdf`：loss、验证指标、学习率和梯度范数四幅图；虚线标记选中的 epoch。
- `history.jsonl`：训练期间持续追加的原始记录。
- `training_history.csv`、`validation_history.csv`：用于自行绘图或分析的表格。
- `selection.json`：验证集选出的轮次和检查点。
- `checkpoint-125/`、`checkpoint-250/`、`checkpoint-375/`：各轮检查点。
- `eval_validation_step_*_summary.json` 和对应的预测文件：每轮验证指标与逐条输出。

绘图脚本可单独运行，也可以在已有训练日志时生成当前进度图：

```bash
.venv/bin/python scripts/plot_tnews.py outputs/tnews_qwen15b_lora_3epochs_20261001
```

新增绘图依赖 `matplotlib==3.10.6` 已列入 `requirements.txt`。
两步试跑会一并验证选模、保存/加载以及曲线导出流程。

### 首次单轮实验结果（2026-09-30）

`outputs/tnews_qwen15b_lora_20260930/` 已完成两步试跑后的正式训练和评估，
保存的适配器已重新加载验证。实际训练 125 步、一轮，耗时 155.9 秒（不含下载、加载和评估）。
可训练参数 9,232,384，占总参数 0.5945%；PyTorch 记录的训练峰值分配显存为 3.95 GiB，
该数字不含 CUDA 上下文及缓存分配器保留但未使用的显存。

| 指标 | 原始模型 | 微调后 |
| --- | --- | --- |
| 本地测试准确率（300 条，严格格式） | 39.33%（118/300） | 56.33%（169/300） |
| JSON 格式及类别合法率 | 84.00% | 99.33% |
| Macro-F1 | 0.3471 | 0.5119 |
| 验证集 loss（200 条） | 0.5336 | 0.2029 |

准确率提高 17 个百分点；70 条由错变对，19 条由对变错。
这次实验表明模型更适应本任务的标签和输出要求，但仍有 131 条测试样本未通过严格判定。
结果仅针对本次 300 条测试子集，不代表官方排行榜成绩或通用能力提升。
详细结果见 `metrics.json`，逐条对比见 `comparison.json`。

数据来自 [CLUE 官方脚本指定的 TNEWS 下载地址](https://storage.googleapis.com/cluebenchmark/tasks/tnews_public.zip)，
来源可核对 [CLUE 官方训练脚本](https://github.com/CLUEbenchmark/CLUE/blob/master/baselines/models/bert/run_classifier_tnews.sh)。
`scripts/prepare_tnews.py` 对标题做 Unicode/空白归一化、去重，按类别原始比例抽样：

- 2,000 条训练数据和 200 条验证数据取自官方 train，互不重叠。
- 300 条本地测试数据取自官方有标签的 dev，且与上述数据无重复标题。
- 本地测试不是官方排行榜测试；没有做语义近似去重，也无法排除模型预训练时见过公开数据。
- 数据中类别分布不均衡，股票类仅有极少量样本，因此小测试集的分项指标波动较大。

原始文件在 `data/raw/`，处理后的数据、来源哈希和类别统计在 `data/tnews/`。
模型保存在 `models/Qwen2.5-1.5B-Instruct/`，`download_manifest.json` 记录确切版本。

项目 `.venv` 使用 Python 3.10，通过 `--system-site-packages` 复用 `unidomain`
环境的 PyTorch 2.5.1+cu121；微调依赖安装在项目内，不修改该 Conda 环境。
它依赖该原环境继续存在。实际版本清单保存在 `logs/environment_freeze.txt`。

在 GPU 节点可运行：

```bash
bash scripts/run_tnews.sh tnews_qwen15b_lora_20260930
```

脚本先跑两步小样本检查，再正式训练一轮（batch 4、梯度累积 4，共 125 次参数更新）。
初始学习率 2e-4，序列上限 256；遇到超长样本会报错，避免静默截掉答案。
每 50 步保存检查点，保留最近两个。复跑请换实验名，避免覆盖结果。

当前终端 `head1` 是登录节点，不能直接做 GPU 训练。2026-09-30 检查到作业
`223699` 在 `node12` 分配了 1 张 A40（约 45 GiB 显存），可在其仍有效时运行：

```bash
srun --jobid=223699 --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --immediate=15 bash scripts/run_tnews.sh tnews_qwen15b_lora_20260930
```

在 `outputs/实验名/` 查看：

- `status.json`：当前阶段；`trainer_state.json`：训练日志。
- `run_config.json`：模型版本、数据哈希、参数、GPU 和软件版本。
- `before_metrics.json`、`after_metrics.json`、`metrics.json`：相同测试集的前后准确率、macro-F1、严格 JSON 格式合规率，以及验证损失。
- `comparison.json`：逐条标题、标准答案及训练前后的模型输出。
- `adapter/`：LoRA 权重；正式训练后的评估会重新加载这个目录，验证保存和加载流程。
- `train_data.jsonl`、`validation_data.jsonl`、`test_data.jsonl`：本次实际用到的数据。

推理示例（在 GPU 节点执行；去掉 `--adapter` 可使用原始模型）：

```bash
.venv/bin/python scripts/predict_tnews.py '国家队公布新一期集训名单' --adapter outputs/tnews_qwen15b_lora_20260930/adapter
```

判断效果时同时看独立测试准确率和验证损失，训练损失下降本身不代表泛化变好。
准确率采用严格标准：输出必须是仅含 `category` 的合法 JSON，类别属于规定的 15 类且与标签一致；格式不合规也计为错误。
这个实验学习的是新闻分类任务，不能据此声称通用聊天能力提高。

## 目录

- `data/`：训练集、验证集及数据处理结果。
- `configs/`：模型与训练配置。
- `scripts/`：数据处理、训练和评估脚本。
- `notebooks/`：交互探索。
- `outputs/`：模型权重、检查点及评估结果。
- `logs/`：实验日志。

## 第一个案例：Qwen 问答模型的 LoRA 微调

参考 [Hugging Face 官方 TRL SFT 教程](https://huggingface.co/docs/trl/v0.21.0/en/sft_trainer)，
将示例模型替换为 [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)，
使用教程中的 [Capybara 对话数据](https://huggingface.co/datasets/trl-lib/Capybara)。
这是对官方案例的小规模改编，不是官方公布的效果复现。

你会依次体验：加载模型 → 准备数据 → 查看训练前回答 → 微调 → 对比验证损失和回答 → 保存权重。
初次练习的目标是跑通流程；模型本身已经会对话，少量训练不保证能力提高。
Capybara 以英文对话为主，本练习不是中文能力提升实验。

三个术语：

- **SFT（监督微调）**：用示例对话训练模型预测接下来的词元。
- **LoRA**：冻结原始模型参数，只训练附加的小矩阵，减少需要训练和保存的参数。
- **Adapter（适配器）**：本次训练产生的小权重文件，使用时仍需要原始模型。

当前机器检测到一张 A40，约 45 GiB 可见显存。这里采用 0.5B 模型、BF16、单卡 LoRA、
每批 1 条、最长 512 个词元，预计适合该显卡；实际显存和耗时以试跑为准。
为便于入门，损失计算覆盖整个对话（包含用户文本），超过 512 个词元的部分截断。
这与仅学习助手回答的训练方式不同。

## 1. 创建独立环境

以下命令在终端逐条执行，首次安装和运行需要访问软件源及 Hugging Face。
固定版本是为了与这个示例对应，不代表最新版本。
PyTorch 使用官方提供的 CUDA 12.1 构建，不需要先升级系统 CUDA。

```bash
cd /cluster/home/sjtu_bkb/LLM-Finetuning
conda create -n llm-finetuning python=3.10 -y
conda activate llm-finetuning
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

最后应能看到 `True` 和 `NVIDIA A40`。
安装来源：[PyTorch 官方历史版本](https://pytorch.org/get-started/previous-versions/)。

## 2. 先跑两步，检查流程

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_lora.py --steps 2 --samples 32 --eval-samples 8 --output outputs/smoke_2steps
```

第一次会下载模型和数据，下载耗时不属于训练耗时。
脚本在终端打印可训练参数数量和每步损失；这两步只检查流程，不用于判断效果。

## 3. 跑一个 50 步练习

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_lora.py --steps 50 --samples 1000 --eval-samples 100 --output outputs/qwen_lora_50steps
```

每次更新累计 8 个小批次，所以 50 步约处理 400 条训练样本，并不是完整训练 1000 条一遍。
脚本先去除完全相同的对话，再固定随机种子划分互不重叠的训练和验证子集。
近似重复并未消除；这不是严格的基准测试。
再次运行时更换 `--output`，避免覆盖已有实验。

## 4. 看结果

- `outputs/qwen_lora_50steps/comparison.json`：同样的三个问题，训练前后各回答一次，使用确定性解码。
- `outputs/qwen_lora_50steps/metrics.json`：相同验证集上的训练前后损失，以及训练统计。
- `outputs/qwen_lora_50steps/trainer_state.json`：逐步训练日志。
- `outputs/qwen_lora_50steps/adapter/`：LoRA 权重、适配器配置和 tokenizer。
- `outputs/qwen_lora_50steps/train_data.jsonl` 和 `eval_data.jsonl`：本次实际选取的数据。
- `outputs/qwen_lora_50steps/run_config.json`：主要实验参数。

先观察 loss 是否有限、有没有 NaN，再比较验证集 loss 和实际回答。
训练 loss 下降不等于通用能力提升，三个问题也不足以证明模型整体变强。

## 5. 重新加载微调结果

在同一个环境中运行以下 Python 代码（或保存为自己的推理脚本）：

```python
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

path = "outputs/qwen_lora_50steps/adapter"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoPeftModelForCausalLM.from_pretrained(
    path, torch_dtype=torch.bfloat16, use_safetensors=True,
).to("cuda").eval()
text = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Give three practical tips for learning Python."}],
    tokenize=False, add_generation_prompt=True,
)
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda")
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=128, do_sample=False, use_cache=True)
print(tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

LoRA 保存与加载说明：[PEFT 官方入门](https://huggingface.co/docs/peft/v0.17.0/en/quicktour)。

## 旧版 Capybara 示例状态

本节的 0.5B + Capybara 示例已准备脚本与操作说明，但尚未下载其模型和数据或执行训练；
当前实际执行的 TNEWS 实验见文档开头。
静态检查不代表训练成功。先执行两步试跑，确认实际依赖、网络和 GPU 正常。

也可在支持的编辑器中打开 `LLM-Finetuning.code-workspace`。
