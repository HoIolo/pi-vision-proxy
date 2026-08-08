import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, extname, join } from "node:path";
import { uuidv7, type ImageContent } from "@earendil-works/pi-ai";
import { complete, type Message } from "@earendil-works/pi-ai/compat";
import { getAgentDir, type ExtensionAPI, type ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

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
// mimo 等模型有 reasoning 行为, token 上限过小时 content 可能为空; 4096 保证描述质量
const VISION_MAX_TOKENS = 4096;
const MAX_CONCURRENT = 4;
// 超过该大小的图片跳过识别(同步 base64 会卡 UI), 并提示用户
const MAX_IMAGE_BYTES = 15 * 1024 * 1024;
const NOTIFY_DEDUP_MS = 10_000;

const DEFAULT_VISION_PROMPT = `你是多模态辅助模型, 代替只支持文本的主模型处理用户发送的图片。
用户消息: {userText}
请优先按用户消息的意图处理图片, 直接输出处理结果(不要任何前缀或解释):
- 如果用户要求参考/模仿图片(如"参考这张图的样式/风格/布局/配色做..."), 请提取主模型复现所需的一切细节: 配色(尽量给具体色值)、字体、布局结构、间距、组件/元素清单、风格特征、图片/图标等, 描述到主模型不看图也能复现的程度
- 如果用户有具体问题(如"这是什么""翻译图中文字""读取报错信息""图中数据"), 直接回答问题
- 如果用户没有具体要求, 则详细描述图片内容: 所有可见文字原样列出、界面布局、元素及其位置
- 如果一次提供了多张图片: 先整体分析(对比差异、共同点、相互关系), 再按图片顺序逐张说明关键内容
如果图片与用户消息不相关(如用户只是在聊天中附带图片), 简要说明图片内容即可, 不要编造细节。`;

// 渲染自定义模板: {userText} 替换为用户消息(可能为空)
function renderPrompt(template: string, userText: string): string {
  const text = userText.trim() || "(用户没有额外说明, 请按最后一条规则处理)";
  return template.replaceAll("{userText}", text);
}

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

/** 读取本地图片文件为 ImageContent; 不存在/格式不支持/超过大小上限返回 undefined。 */
function readImageFile(filePath: string): ImageContent | undefined {
  try {
    const mimeType = MIME_BY_EXT[extname(filePath).toLowerCase()];
    if (!mimeType) return undefined;
    if (statSync(filePath).size > MAX_IMAGE_BYTES) return undefined; // 大图跳过, 防同步 base64 卡 UI
    const data = readFileSync(filePath).toString("base64");
    return { type: "image", data, mimeType };
  } catch {
    return undefined;
  }
}

interface VisionConfig {
  visionModel: string;
  /** 自定义识图 prompt 模板, {userText} 为用户消息占位符; 省略用默认 */
  prompt?: string;
}

function readConfig(): VisionConfig {
  try {
    const parsed = JSON.parse(readFileSync(CONFIG_FILE, "utf8")) as Partial<VisionConfig>;
    const config: VisionConfig = { visionModel: DEFAULT_VISION_MODEL };
    if (typeof parsed.visionModel === "string" && parsed.visionModel.trim()) {
      config.visionModel = parsed.visionModel.trim();
    }
    if (typeof parsed.prompt === "string" && parsed.prompt.trim()) {
      config.prompt = parsed.prompt.trim();
    }
    return config;
  } catch {
    // 配置文件不存在或损坏: 使用默认值
    return { visionModel: DEFAULT_VISION_MODEL };
  }
}

function writeConfig(config: VisionConfig): void {
  mkdirSync(dirname(CONFIG_FILE), { recursive: true });
  writeFileSync(CONFIG_FILE, JSON.stringify(config, null, 2), "utf8");
}

// 结果缓存 + 单飞去重: 同一张图片(按内容哈希)在同一识图模型、同一 prompt 下只识别一次;
// prompt 不同视为不同请求(允许对同一张图深入查看/换角度提问)
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

function cacheKey(visionModel: string, image: ImageContent, prompt = ""): string {
  const hash = createHash("sha1").update(image.data).digest("hex");
  const promptHash = createHash("sha1").update(prompt).digest("hex").slice(0, 12);
  return `${visionModel}:${hash}:${promptHash}`;
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
    const { prompt: customPrompt } = readConfig();
    const template = customPrompt?.trim() || DEFAULT_VISION_PROMPT;
    const prompt = renderPrompt(template, userText);
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
  const key = cacheKey(visionModel, image, userText);
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

/** 多图合并缓存 key: 所有图哈希(按序) + prompt。 */
function multiCacheKey(
  visionModel: string,
  images: ImageContent[],
  prompt = "",
): string {
  const imgHash = images
    .map((i) => createHash("sha1").update(i.data).digest("hex").slice(0, 12))
    .join("+");
  const promptHash = createHash("sha1").update(prompt).digest("hex").slice(0, 12);
  return `multi:${visionModel}:${imgHash}:${promptHash}`;
}

/** 一次调用识别多张图(支持对比/关系类问题); 任何失败返回 undefined。 */
async function runVisionMulti(
  ctx: ExtensionContext,
  visionModel: string,
  images: ImageContent[],
  userText: string,
  key: string,
): Promise<string | undefined> {
  const model = resolveModel(ctx, visionModel);
  if (!model || !model.input?.includes("image")) return undefined;

  const auth = await ctx.modelRegistry.getApiKeyAndHeaders(model);
  if (!auth.ok || !auth.apiKey) return undefined;

  const timeout = AbortSignal.timeout(VISION_TIMEOUT_MS);
  const signal = ctx.signal ? AbortSignal.any([ctx.signal, timeout]) : timeout;

  try {
    const { prompt: customPrompt } = readConfig();
    const template = customPrompt?.trim() || DEFAULT_VISION_PROMPT;
    const prompt = renderPrompt(template, userText);
    const content: Array<{ type: "text"; text: string } | ImageContent> = [
      { type: "text", text: prompt },
    ];
    for (const img of images) {
      content.push({ type: "image", data: img.data, mimeType: img.mimeType });
    }
    const messages: Message[] = [
      { role: "user", content, timestamp: Date.now() },
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

/**
 * 仅查缓存(不触发识别): 返回已识别过的描述数组; 未识别过返回 undefined。
 * 多图: 优先合并缓存, 回退逐图缓存(合并失败时逐张识别的结果)。
 */
function cachedDescs(
  visionModel: string,
  images: ImageContent[],
  userText: string,
): Array<string | undefined> | undefined {
  if (images.length === 1) {
    const k = cacheKey(visionModel, images[0], userText);
    if (!descCache.has(k)) return undefined;
    return [descCache.get(k)];
  }
  const mk = multiCacheKey(visionModel, images, userText);
  const merged = descCache.get(mk);
  if (merged !== undefined) {
    const out: Array<string | undefined> = [merged];
    for (let i = 1; i < images.length; i++) out.push("(已包含在上方合并识别结果中)");
    return out;
  }
  const out: Array<string | undefined> = [];
  for (const img of images) {
    const k = cacheKey(visionModel, img, userText);
    if (!descCache.has(k)) return undefined;
    out.push(descCache.get(k));
  }
  return out;
}

/**
 * 多图识别: 多图时合并为一次调用(整体分析 + 对比), 结果挂在第一张图位置;
 * 合并失败回退为逐张并行识别(限流 MAX_CONCURRENT)。返回与输入顺序一致的结果数组。
 */
async function describeImages(
  ctx: ExtensionContext,
  visionModel: string,
  images: ImageContent[],
  userText: string,
): Promise<Array<string | undefined>> {
  if (images.length === 1) {
    return [await describeImage(ctx, visionModel, images[0], userText)];
  }
  // 多图: 一次合并调用(支持对比/关系问题)
  const key = multiCacheKey(visionModel, images, userText);
  let text = descCache.get(key);
  if (text === undefined) {
    const pending = inFlight.get(key);
    if (pending) {
      text = await pending;
    } else {
      const p = runVisionMulti(ctx, visionModel, images, userText, key);
      inFlight.set(key, p);
      try {
        text = await p;
      } finally {
        if (inFlight.get(key) === p) inFlight.delete(key);
      }
    }
  }
  if (text !== undefined) {
    const out = new Array<string | undefined>(images.length);
    out[0] = text;
    for (let i = 1; i < images.length; i++) out[i] = "(已包含在上方合并识别结果中)";
    return out;
  }
  // 合并失败: 回退逐张并行识别
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
 * 超过大小上限的图片跳过并提示。
 */
function collectImagePaths(ctx: ExtensionContext, text: string): ImageContent[] {
  const images: ImageContent[] = [];
  for (const m of text.matchAll(IMAGE_PATH_RE)) {
    const path = m[2];
    if (!existsSync(path)) continue; // 文件不存在/不可读: 忽略
    try {
      if (statSync(path).size > MAX_IMAGE_BYTES) {
        notifyOnce(ctx, `big:${path}`, `图片超过 ${Math.round(MAX_IMAGE_BYTES / 1024 / 1024)}MB 已跳过识别: ${path}`);
        continue;
      }
    } catch {
      continue;
    }
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
    const pathImages = collectImagePaths(ctx, userText);
    // 2) 合并真正的图片块
    const images = [...pathImages, ...(event.images ?? [])];
    if (images.length === 0) return;

    // 后台预热识别, 不阻塞发送; 完成后清除提示(不占用底部状态栏)
    ctx.ui.setWorkingMessage(`识别图片中... (${visionModel})`);
    void describeImages(ctx, visionModel, images, userText).finally(() => {
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
    // 收集历史中所有含图片块的消息
    const targets: Array<{
      msg: (typeof event.messages)[number];
      images: ImageContent[];
      userText: string;
    }> = [];
    for (const m of event.messages) {
      if (
        m.role !== "user" ||
        typeof m.content === "string" ||
        !m.content.some((c) => c.type === "image")
      ) {
        continue;
      }
      const images = m.content.filter((c): c is ImageContent => c.type === "image");
      const userText = m.content
        .filter((c) => c.type === "text")
        .map((c) => c.text)
        .join("\n");
      targets.push({ msg: m, images, userText });
    }
    if (targets.length === 0) return;

    // 仅等待“识别进行中”的图片(当前回合 input 预热的请求); 历史旧图不触发识别, 只放占位提示
    let needsWait = false;
    for (const t of targets) {
      if (cachedDescs(visionModel, t.images, t.userText) !== undefined) continue;
      if (inFlight.has(multiCacheKey(visionModel, t.images, t.userText))) {
        needsWait = true;
        continue;
      }
      for (const img of t.images) {
        if (inFlight.has(cacheKey(visionModel, img, t.userText))) {
          needsWait = true;
          break;
        }
      }
    }
    if (needsWait) {
      ctx.ui.setWorkingMessage(`识别图片中... (${visionModel})`);
      try {
        // 等待所有进行中的识别完成(并发有限)
        await Promise.allSettled([...inFlight.values()].map((p) => p.catch(() => undefined)));
      } finally {
        ctx.ui.setWorkingMessage();
      }
    }

    const messages = event.messages.map((m) => {
      const ti = targets.findIndex((t) => t.msg === m);
      if (ti < 0) return m;
      const t = targets[ti];
      const descs = cachedDescs(visionModel, t.images, t.userText);
      const newContent = [];
      let imgIdx = 0;
      for (const block of m.content) {
        if (block.type === "image") {
          const desc = descs?.[imgIdx++];
          if (desc) {
            newContent.push({ type: "text", text: `[图片识别结果 - ${visionModel}]\n${desc}` });
          } else {
            // 未识别过的图片: 原样保留, 不自动识别不占位(pi 对 text-only 模型自动省略,
            // 主模型需要时可调用 vision_describe 工具按需查看)
            newContent.push(block);
          }
        } else {
          newContent.push(block);
        }
      }
      return { ...m, content: newContent };
    });
    return { messages };
  });

  const VISION_TOOL_NAME = "vision_describe";

  // 动态注册: 当前模型多模态时从 active 工具中移除 vision_describe(系统 prompt 与模型 tools API 同步消失),
  // text-only 时加回。跟随模型切换实时生效。
  function syncVisionTool(model: { input?: string[] } | undefined): void {
    if (!model) return;
    const supportsImage = model.input?.includes("image") ?? false;
    const active = pi.getActiveTools();
    const has = active.includes(VISION_TOOL_NAME);
    if (supportsImage && has) {
      pi.setActiveTools(active.filter((t) => t !== VISION_TOOL_NAME));
    } else if (!supportsImage && !has) {
      pi.setActiveTools([...active, VISION_TOOL_NAME]);
    }
  }
  pi.on("model_select", (event) => {
    syncVisionTool(event.model);
  });
  pi.on("session_start", (_event, ctx) => {
    syncVisionTool(ctx.model);
  });

  // 多模态主模型下禁用 vision_describe: 拦截调用(pi 扩展 API 无工具注销机制, 用运行时拦截达到等效效果)
  pi.on("tool_call", async (event, ctx) => {
    if (event.toolName !== "vision_describe") return;
    const model = ctx.model;
    if (model?.input?.includes("image")) {
      return {
        block: true,
        reason: `当前模型 ${model.provider}/${model.id} 原生支持图片输入, 不需要 vision_describe 工具, 请直接查看图片`,
      };
    }
  });

  // 主模型可主动调用的识图工具: 需要查看任意本地图片时自行调用, 可携带处理要求
  pi.registerTool({
    name: "vision_describe",
    label: "识图",
    description:
      "识别/处理本地图片(调用独立的多模态识图模型, 不消耗主模型多模态能力)。当主模型只支持文本、需要查看图片内容时调用, 例如用户消息中提到的图片文件、工具生成或发现的截图等, 也可对已看过的图片提出更具体的问题深入查看。参数 path 为本地图片文件路径(支持 png/jpg/jpeg/gif/webp/bmp, 超过 15MB 跳过); 需要对比/分析多张图时, 用 paths 传入其余图片路径, 一次调用整体处理; 如有具体需求(提取配色、翻译图中文字、读取数据、描述布局、判断内容等)请通过 prompt 参数传入, 识图模型会按你的要求处理, 效果等同原生多模态模型看图。",
    promptSnippet: "vision_describe(path, prompt?, paths?) - 用多模态识图模型处理本地图片, 可按需传入处理要求与多图对比",
    parameters: Type.Object({
      path: Type.String({ description: "本地图片文件路径" }),
      prompt: Type.Optional(
        Type.String({ description: "对图片的处理要求(可选): 如提取配色、翻译图中文字、读取表格数据、描述布局风格、对比多张图等; 省略时默认详细描述图片内容" }),
      ),
      paths: Type.Optional(
        Type.Array(Type.String({ description: "附加图片路径(可选): 与 path 一起一次调用综合处理, 适合对比/关系类问题" })),
      ),
    }),
    async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
      const { visionModel } = readConfig();
      ctx.ui.setWorkingMessage(`识别图片中... (${visionModel})`);
      try {
        const allPaths = [params.path, ...(params.paths ?? [])];
        const images: ImageContent[] = [];
        const failed: string[] = [];
        for (const p of allPaths) {
          const img = readImageFile(p);
          if (img) images.push(img);
          else failed.push(p);
        }
        if (images.length === 0) {
          return {
            content: [
              {
                type: "text" as const,
                text: `无法读取图片(文件不存在、不可读、超过 15MB 或格式不支持): ${failed.join(", ")}`,
              },
            ],
            details: {},
          };
        }
        const descs = await describeImages(ctx, visionModel, images, params.prompt ?? "");
        const parts = descs.filter((d): d is string => !!d);
        const main = parts.length > 0 ? parts.join("\n") : "[图片识别失败, 已跳过; 可稍后重试]";
        const note = failed.length > 0 ? `\n(以下图片读取失败已跳过: ${failed.join(", ")})` : "";
        return {
          content: [{ type: "text" as const, text: main + note }],
          details: {},
        };
      } catch (err) {
        return {
          content: [
            {
              type: "text" as const,
              text: `[图片识别出错: ${err instanceof Error ? err.message : String(err)}]`,
            },
          ],
          details: {},
        };
      } finally {
        ctx.ui.setWorkingMessage();
      }
    },
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
