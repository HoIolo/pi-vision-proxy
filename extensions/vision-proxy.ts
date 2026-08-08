import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, extname, join } from "node:path";
import { uuidv7, type ImageContent } from "@earendil-works/pi-ai";
import { complete, type Message } from "@earendil-works/pi-ai/compat";
import { getAgentDir, type ExtensionAPI, type ExtensionContext } from "@earendil-works/pi-coding-agent";

/**
 * 图片识别代理(优化版):
 * - pi 粘贴图片(Ctrl+V)时插入的是临时文件路径文本, 不是图片块。
 *   因此 input 事件里检测文本中的图片路径: 读取文件 → 识图模型描述 →
 *   transform 替换为描述文本, agent 直接看到描述, 无需自行读图。
 * - 真正的图片块(images 数组, 如 RPC/SDK 发送): input 事件后台预热识图(不阻塞),
 *   context 事件时通常已完成; context 事件里将图片块替换为描述文本。
 * - 多图并行识别(限流 4), 单图 45s 超时, 相同图片单飞去重 + 结果缓存
 * - 识别失败: 替换为占位文本(对 text-only 模型保留图片只会导致主调用失败)
 * - 当前模型支持 image 时不做任何处理(模型原生看图)
 * - 识图模型可通过 /vision-model <provider/model> 设置(仅当前模型 text-only 时允许),
 *   配置保存在 ~/.pi/agent/vision-proxy.json
 */

const CONFIG_FILE = join(getAgentDir(), "vision-proxy.json");
const DEFAULT_VISION_MODEL = "opencode/mimo-v2.5-free";
const CACHE_LIMIT = 50;
const VISION_TIMEOUT_MS = 45_000;
const VISION_MAX_TOKENS = 1024;
const MAX_CONCURRENT = 4;
const NOTIFY_DEDUP_MS = 10_000;

const VISION_PROMPT = `你是图像识别助手, 代替只支持文本的主模型识别用户发送的图片。请仔细观察这张图片, 并尽可能详细地描述其中的内容:
- 所有可见的文字请原样列出(包括代码、报错信息、UI 文案、图表数据、链接等)
- 描述图片中的界面、布局、元素及其位置关系
- 如果用户消息中包含针对图片的问题, 请优先回答该问题
直接输出描述内容, 不要任何前缀或解释。`;

// 匹配消息文本中出现的本地图片路径(pi 粘贴图片时插入的是临时文件路径文本)
const IMAGE_PATH_RE =
  /(^|[\s"'`])([^\s"'`]+\.(?:png|jpe?g|gif|webp|bmp))(?=$|[\s"'`,;])/gi;

const MIME_BY_EXT: Record<string, string> = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".webp": "image/webp",
  ".bmp": "image/bmp",
};

/** 读取本地图片文件为 ImageContent; 失败返回 undefined。 */
function readImageFile(filePath: string): ImageContent | undefined {
  try {
    const mimeType = MIME_BY_EXT[extname(filePath).toLowerCase()];
    if (!mimeType) return undefined;
    const data = readFileSync(filePath).toString("base64");
    return { type: "image", data, mimeType };
  } catch {
    return undefined;
  }
}

interface VisionConfig {
  visionModel: string;
}

function readConfig(): VisionConfig {
  try {
    const parsed = JSON.parse(readFileSync(CONFIG_FILE, "utf8")) as Partial<VisionConfig>;
    if (typeof parsed.visionModel === "string" && parsed.visionModel.trim()) {
      return { visionModel: parsed.visionModel.trim() };
    }
  } catch {
    // 配置文件不存在或损坏: 使用默认值
  }
  return { visionModel: DEFAULT_VISION_MODEL };
}

function writeConfig(config: VisionConfig): void {
  mkdirSync(dirname(CONFIG_FILE), { recursive: true });
  writeFileSync(CONFIG_FILE, JSON.stringify(config, null, 2), "utf8");
}

// 结果缓存 + 单飞去重: 同一张图片(按内容哈希)在同一识图模型下只识别一次
const descCache = new Map<string, string>();
const inFlight = new Map<string, Promise<string | undefined>>();

// 失败通知去重, 避免预热/重试时刷屏
let lastFailureNotify = { key: "", at: 0 };
function notifyOnce(ctx: ExtensionContext, key: string, message: string): void {
  const now = Date.now();
  if (key === lastFailureNotify.key && now - lastFailureNotify.at < NOTIFY_DEDUP_MS) return;
  lastFailureNotify = { key, at: now };
  if (ctx.hasUI) ctx.ui.notify(message, "warning");
}

function cacheKey(visionModel: string, image: ImageContent): string {
  const hash = createHash("sha1").update(image.data).digest("hex");
  return `${visionModel}:${hash}`;
}

function parseModelRef(ref: string): { provider?: string; id: string } {
  const idx = ref.indexOf("/");
  return idx > 0
    ? { provider: ref.slice(0, idx), id: ref.slice(idx + 1) }
    : { id: ref };
}

function resolveModel(ctx: ExtensionContext, ref: string) {
  const { provider, id } = parseModelRef(ref);
  if (provider) return ctx.modelRegistry.find(provider, id);
  return ctx.modelRegistry.getAll().find((m) => m.id === id);
}

/** 实际调用识图模型; 任何失败返回 undefined。 */
async function runVision(
  ctx: ExtensionContext,
  visionModel: string,
  image: ImageContent,
  userText: string,
  key: string,
): Promise<string | undefined> {
  const model = resolveModel(ctx, visionModel);
  if (!model || !model.input?.includes("image")) return undefined;

  const auth = await ctx.modelRegistry.getApiKeyAndHeaders(model);
  if (!auth.ok || !auth.apiKey) return undefined;

  // 合并调用方 abort + 超时, 避免识图卡死整个回合
  const timeout = AbortSignal.timeout(VISION_TIMEOUT_MS);
  const signal = ctx.signal ? AbortSignal.any([ctx.signal, timeout]) : timeout;

  try {
    const prompt = userText.trim()
      ? `${VISION_PROMPT}\n\n用户消息: ${userText.trim()}`
      : VISION_PROMPT;
    const messages: Message[] = [
      {
        role: "user",
        content: [
          { type: "text", text: prompt },
          { type: "image", data: image.data, mimeType: image.mimeType },
        ],
        timestamp: Date.now(),
      },
    ];
    const response = await complete(model, { messages }, {
      apiKey: auth.apiKey,
      headers: auth.headers,
      env: auth.env,
      signal,
      maxTokens: VISION_MAX_TOKENS,
      cacheRetention: "none",
      sessionId: uuidv7(),
    });
    const text = response.content
      .filter((c): c is { type: "text"; text: string } => c.type === "text")
      .map((c) => c.text)
      .join("\n")
      .trim();
    if (!text) return undefined;
    if (descCache.size >= CACHE_LIMIT) {
      const oldest = descCache.keys().next().value;
      if (oldest !== undefined) descCache.delete(oldest);
    }
    descCache.set(key, text);
    return text;
  } catch (error) {
    const reason = error instanceof Error && error.name === "TimeoutError"
      ? "超时"
      : error instanceof Error ? error.message : String(error);
    notifyOnce(ctx, key, `图片识别失败(${visionModel}): ${reason}`);
    return undefined;
  }
}

/** 单飞入口: 命中缓存直接返回; 已有同图请求则共享; 否则启动新请求。 */
function describeImage(
  ctx: ExtensionContext,
  visionModel: string,
  image: ImageContent,
  userText: string,
): Promise<string | undefined> {
  const key = cacheKey(visionModel, image);
  const cached = descCache.get(key);
  if (cached !== undefined) return Promise.resolve(cached);
  const pending = inFlight.get(key);
  if (pending) return pending;
  const p = runVision(ctx, visionModel, image, userText, key);
  inFlight.set(key, p);
  void p.finally(() => {
    if (inFlight.get(key) === p) inFlight.delete(key);
  });
  return p;
}

/** 多图并行识别(限流 MAX_CONCURRENT), 返回与输入顺序一致的结果数组。 */
async function describeImages(
  ctx: ExtensionContext,
  visionModel: string,
  images: ImageContent[],
  userText: string,
): Promise<Array<string | undefined>> {
  const out = new Array<string | undefined>(images.length);
  for (let i = 0; i < images.length; i += MAX_CONCURRENT) {
    const slice = images.slice(i, i + MAX_CONCURRENT);
    const results = await Promise.all(
      slice.map((img) => describeImage(ctx, visionModel, img, userText)),
    );
    results.forEach((r, j) => {
      out[i + j] = r;
    });
  }
  return out;
}

/**
 * 扫描文本中的本地图片路径(pi 粘贴图片的默认形式), 将存在的图片转为图片块。
 * 不修改原文本: 路径保留在消息中, 方便用户回查传过什么图片。
 */
function collectImagePaths(text: string): ImageContent[] {
  const images: ImageContent[] = [];
  for (const m of text.matchAll(IMAGE_PATH_RE)) {
    const path = m[2];
    if (!existsSync(path)) continue; // 文件不存在/不可读: 忽略
    const image = readImageFile(path);
    if (image) images.push(image);
  }
  return images;
}

export default function (pi: ExtensionAPI) {
  // 用户提交消息时立即处理:
  // - 文本中的图片路径 → 转为图片块(消息保留图片, 交互与多模态模型一致), 并后台预热识图
  // - 真正的图片块 → 后台预热识图(context 事件时大概率已完成, 直接替换为描述)
  pi.on("input", async (event, ctx) => {
    const model = ctx.model;
    if (!model || model.input?.includes("image")) return; // 主模型能看图则无需代理
    const { visionModel } = readConfig();
    const userText = event.text ?? "";

    // 1) 文本中的本地图片路径(pi 粘贴图片的默认形式)
    const pathImages = collectImagePaths(userText);
    // 2) 合并真正的图片块
    const images = [...pathImages, ...(event.images ?? [])];
    if (images.length === 0) return;

    // 后台预热识别, 不阻塞发送; 完成后清除提示
    ctx.ui.setStatus("vision-proxy", `识别图片中... (${visionModel})`);
    ctx.ui.setWorkingMessage(`识别图片中... (${visionModel})`);
    void describeImages(ctx, visionModel, images, userText).finally(() => {
      ctx.ui.setStatus("vision-proxy", undefined);
      ctx.ui.setWorkingMessage();
    });

    // 有路径转换: transform 附加图片块, 文本原样保留(含路径, 便于回查), 消息在 UI 上保留图片缩略图
    if (pathImages.length > 0) {
      return { action: "transform" as const, text: userText, images };
    }
  });

  pi.on("context", async (event, ctx) => {
    const model = ctx.model;
    if (!model) return;
    // 当前模型支持图片: 原生处理, 不代理
    if (model.input?.includes("image")) return;

    const { visionModel } = readConfig();
    let changed = false;
    const messages = [];
    for (const m of event.messages) {
      if (
        m.role !== "user" ||
        typeof m.content === "string" ||
        !m.content.some((c) => c.type === "image")
      ) {
        messages.push(m);
        continue;
      }
      const images = m.content.filter((c): c is ImageContent => c.type === "image");
      const userText = m.content
        .filter((c) => c.type === "text")
        .map((c) => c.text)
        .join("\n");

      ctx.ui.setStatus("vision-proxy", `识别图片中... (${visionModel})`);
      ctx.ui.setWorkingMessage(`识别图片中... (${visionModel})`);
      let descs: Array<string | undefined>;
      try {
        descs = await describeImages(ctx, visionModel, images, userText);
      } finally {
        ctx.ui.setStatus("vision-proxy", undefined);
        ctx.ui.setWorkingMessage();
      }

      const newContent = [];
      let imgIdx = 0;
      for (const block of m.content) {
        if (block.type === "image") {
          const desc = descs[imgIdx++];
          newContent.push(
            desc
              ? { type: "text", text: `[图片识别结果 - ${visionModel}]\n${desc}` }
              : { type: "text", text: "[图片识别失败, 已跳过]" },
          );
        } else {
          newContent.push(block);
        }
      }
      messages.push({ ...m, content: newContent });
      changed = true;
    }
    if (!changed) return;
    return { messages };
  });

  pi.registerCommand("vision-model", {
    description:
      "查看/设置图片识别模型(仅当当前模型只支持 text 时可设置; 默认 opencode/mimo-v2.5-free; 用法: !vision-model [provider/model | reset])",
    handler: async (args, ctx) => {
      const arg = args.trim();
      const config = readConfig();
      const current = ctx.model;
      const currentSupportsImage = current?.input?.includes("image") ?? false;

      if (!arg) {
        const native = currentSupportsImage ? " (当前模型原生支持图片, 代理不生效)" : "";
        ctx.ui.notify(`当前识图模型: ${config.visionModel}${native}`, "info");
        // 仅当当前模型只支持 text 时可设置; 有 UI 时弹出交互选择器(只列支持图片的模型)
        if (!currentSupportsImage && ctx.hasUI) {
          const candidates = (ctx.modelRegistry.getAvailable().length > 0
            ? ctx.modelRegistry.getAvailable()
            : ctx.modelRegistry.getAll())
            .filter((m) => m.input?.includes("image"))
            .map((m) => `${m.provider}/${m.id}`)
            .sort();
          if (candidates.length === 0) {
            ctx.ui.notify("没有支持图片的模型可用", "warning");
            return;
          }
          const chosen = await ctx.ui.select("选择识图模型(仅支持图片的模型)", candidates);
          if (!chosen) return; // 用户取消
          writeConfig({ visionModel: chosen });
          ctx.ui.notify(`识图模型已设置为 ${chosen}`, "info");
        } else if (!currentSupportsImage) {
          ctx.ui.notify(`用法: !vision-model <provider/model> 或直接编辑 ${CONFIG_FILE}`, "info");
        }
        return;
      }
      if (arg === "reset") {
        writeConfig({ visionModel: DEFAULT_VISION_MODEL });
        ctx.ui.notify(`已重置识图模型为默认: ${DEFAULT_VISION_MODEL}`, "info");
        return;
      }
      // 仅当当前模型只支持 text 时允许设置识图模型(复用上方已声明的 current)
      if (current?.input?.includes("image")) {
        ctx.ui.notify(
          `当前模型 ${current.provider}/${current.id} 支持图片输入, 不需要设置识图模型`,
          "error",
        );
        return;
      }
      const target = resolveModel(ctx, arg);
      if (!target) {
        ctx.ui.notify(`未找到模型: ${arg}`, "error");
        return;
      }
      if (!target.input?.includes("image")) {
        ctx.ui.notify(`模型 ${target.provider}/${target.id} 不支持图片输入, 不能作为识图模型`, "error");
        return;
      }
      writeConfig({ visionModel: `${target.provider}/${target.id}` });
      ctx.ui.notify(`识图模型已设置为 ${target.provider}/${target.id}`, "info");
    },
  });
}
