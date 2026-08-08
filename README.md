# pi-vision-proxy

Pi 扩展:当主模型只支持文本(text-only)时,自动把图片识别交给一个多模态模型(默认 `opencode/mimo-v2.5-free`),让 text-only 模型也能"看图"。交互体验接近原生多模态模型:粘贴/发送图片后立即发送,消息里保留图片路径和缩略图,识别期间显示"识别图片中..."提示,识别完成后 agent 直接收到图片描述。

## 安装

```bash
# 本地路径
pi install /path/to/pi-vision-proxy

# npm(发布后)
pi install npm:pi-vision-proxy

# git(发布后)
pi install git:github.com/<user>/pi-vision-proxy
```

安装后 `/reload` 生效。

> 注意:如果你之前手动把 `vision-proxy.ts` 放在 `~/.pi/agent/extensions/` 下,请先删除它,避免与包内版本重复注册命令。

## 用法

- **直接发图**:粘贴(Ctrl+V)图片或发送带图片路径的消息。扩展会在后台调用识图模型,把描述注入给主模型。当前模型本身就支持图片时,扩展完全不做处理(模型原生看图)。
- **`/vision-model`**:查看当前识图模型;当前模型为 text-only 时弹出选择器,列出所有支持图片的模型供选择。
- **`/vision-model <provider/model>`**:直接设置,例如 `/vision-model 123nhh/claude-haiku-4-5-20251001`。
- **`/vision-model reset`**:恢复默认 `opencode/mimo-v2.5-free`。

配置保存在 `~/.pi/agent/vision-proxy.json`,也可以直接编辑该文件:

```json
{
  "visionModel": "opencode/mimo-v2.5-free",
  "prompt": "自定义识图 prompt 模板, {userText} 会被替换为用户消息"
}
```

默认 prompt 是**动态**的:识图模型先看用户消息再处理图片——用户要求参考/模仿图片时提取样式细节(配色、布局、组件等),有具体问题时直接回答,没有要求时才做通用描述。

## 工作原理

1. **input 事件**:扫描消息文本中的本地图片路径(pi 粘贴图片插入的是临时文件路径文本),读取为图片块;文本原样保留(含路径,便于回查)。同时**后台预热**识图(不阻塞发送),UI 显示"识别图片中..."。
2. **context 事件**:主模型为 text-only 时,把图片块替换为识图模型生成的描述文本;多图并行识别(限流 4)、单图 45s 超时、相同图片去重 + 缓存。
3. **vision_describe 工具**:主模型在对话过程中需要查看任意本地图片时,可主动调用 `vision_describe(path)` 工具(如用户提到的图片文件、工具生成的截图),无需重新发消息。
4. 主模型支持图片时所有逻辑跳过,行为与普通多模态模型一致。

## 要求

- 需要至少一个支持图片输入(image)的模型可用(可在 `/vision-model` 选择器中查看)。
- 识图模型的 provider 需要有可用的 API 认证(与 pi 其他模型一致)。
- 扩展本身无第三方运行时依赖,仅依赖 pi 核心包。
