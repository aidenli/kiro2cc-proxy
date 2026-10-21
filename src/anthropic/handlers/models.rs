// Copyright (c) 2026 Harllan He. Licensed under MIT.
//! 模型列表 Handler：/v1/models 端点，静态列表与上游动态拉取相结合，带 TTL 缓存（CachedModels）。

use crate::anthropic::middleware::AppState;
use crate::anthropic::types::{ErrorResponse, Model, ModelsResponse};

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json, Response},
};
use std::time::Duration;

/// GET /v1/models
///
/// 返回可用的模型列表（动态来源，带 TTL 缓存 + 静态表回退）
pub async fn get_models(State(state): State<AppState>) -> impl IntoResponse {
    tracing::info!("Received GET /v1/models request");

    Json(ModelsResponse {
        object: "list".to_string(),
        data: fetch_models_dynamic(&state).await,
    })
}

/// 模型缓存别名，简化签名
pub(crate) type ModelCache =
    std::sync::Arc<parking_lot::RwLock<Option<super::super::middleware::CachedModels>>>;

/// 动态获取模型列表（对外唯一出口）：原始结果经 `normalize_model_ids` 归一化后返回。
///
/// `/v1/models` 与 `/v1/models/:model_id` 都经由本函数取数，因此归一化在此处统一生效，
/// 保证两个出口暴露的 ID 集合完全一致（点号 ID 不会在单模型查询里漏出）。
pub(crate) async fn fetch_models_dynamic(state: &AppState) -> Vec<Model> {
    normalize_model_ids(fetch_models_dynamic_raw(state).await)
}

/// 原始动态获取逻辑：优先上游实时响应（带 TTL 缓存），失败按顺序回退。
///
/// 分支：
/// 1. 缓存存在且未过期 → 直接返回缓存，不打上游
/// 2. 缓存过期/缺失且上游成功返回非空模型集 → 映射、写缓存、返回
/// 3. 上游失败但存在旧缓存 → 续用旧缓存（warn）
/// 4. 上游失败且无缓存 → 回退静态表 `build_model_list()`（warn）
/// 5. 未配置 `kiro_provider` → 直接回退静态表（无上游可查）
///
/// 返回值保持上游原始 ID 写法（含点号），归一化由调用方 `fetch_models_dynamic` 负责，
/// 缓存里存的也是原始形态。
async fn fetch_models_dynamic_raw(state: &AppState) -> Vec<Model> {
    // 分支 5：无上游 provider，直接静态表
    let Some(provider) = state.kiro_provider.as_ref() else {
        return build_model_list();
    };

    let ttl = Duration::from_secs(provider.token_manager().config().model_cache_ttl_secs);

    // 分支 1：缓存命中且未过期
    if let Some(hit) = cached_if_fresh(&state.model_cache, ttl) {
        return hit;
    }

    // 缓存缺失/过期，尝试刷新上游（仅此处涉及网络；结果归一化为 Option<Vec<Model>>）
    let refreshed: Option<Vec<Model>> = match provider.token_manager().list_available_models().await
    {
        Ok(resp) if !resp.models.is_empty() => {
            let models: Vec<Model> = resp.models.iter().map(available_model_to_model).collect();
            Some(if models_official_aliases_enabled() {
                with_official_aliases(models)
            } else {
                models
            })
        }
        Ok(_) => {
            tracing::warn!("上游模型列表为空，回退缓存/静态表");
            None
        }
        Err(e) => {
            tracing::warn!(error = %e, "刷新上游模型列表失败，回退缓存/静态表");
            None
        }
    };

    resolve_after_refresh(&state.model_cache, refreshed)
}

/// `/v1/models` 官方连字符别名开关（config `modelsOfficialAliases`，默认 false）
static MODELS_OFFICIAL_ALIASES: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// 设置官方别名开关（main.rs 启动时按 config 调用）
pub fn set_models_official_aliases(enabled: bool) {
    MODELS_OFFICIAL_ALIASES.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

fn models_official_aliases_enabled() -> bool {
    MODELS_OFFICIAL_ALIASES.load(std::sync::atomic::Ordering::Relaxed)
}

/// 4.5 代际在 Anthropic 官方只有带日期的模型 ID（Kiro 点号 ID → 官方 ID）
const DATED_OFFICIAL_IDS: &[(&str, &str)] = &[
    ("claude-sonnet-4.5", "claude-sonnet-4-5-20250929"),
    ("claude-opus-4.5", "claude-opus-4-5-20251101"),
    ("claude-haiku-4.5", "claude-haiku-4-5-20251001"),
];

/// 为上游点号模型 ID 追加 Anthropic 官方连字符别名（紧跟在原条目之后）。
///
/// Kiro `ListAvailableModels` 返回 `claude-opus-4.6` / `claude-opus-5.5` 这类点号 ID，
/// 而客户端与下游网关（如 sub2api 按 /v1/models 生成账号模型映射）使用官方
/// `claude-opus-4-6` / `claude-opus-5-5`。只暴露点号 ID 时，按官方名请求的模型
/// 在网关侧匹配不到本账号。`map_model` 对两种写法映射一致，别名可直接请求。
/// 4.5 代际额外追加带日期的官方 ID；已存在的 ID 不重复追加。
pub(crate) fn with_official_aliases(models: Vec<Model>) -> Vec<Model> {
    let mut seen: std::collections::HashSet<String> = models.iter().map(|m| m.id.clone()).collect();
    let mut out = Vec::with_capacity(models.len() * 2);
    for model in models {
        let mut aliases = Vec::new();
        if model.id.starts_with("claude-") && model.id.contains('.') {
            aliases.push(model.id.replace('.', "-"));
        }
        if let Some((_, dated)) = DATED_OFFICIAL_IDS.iter().find(|(id, _)| *id == model.id) {
            aliases.push((*dated).to_string());
        }
        let aliases: Vec<Model> = aliases
            .into_iter()
            .filter(|id| seen.insert(id.clone()))
            .map(|id| Model {
                id,
                ..model.clone()
            })
            .collect();
        out.push(model);
        out.extend(aliases);
    }
    out
}

/// 缓存命中判定（纯逻辑，无网络）：存在且未超过 TTL 时返回克隆的模型列表。
pub(crate) fn cached_if_fresh(cache: &ModelCache, ttl: Duration) -> Option<Vec<Model>> {
    let guard = cache.read();
    guard
        .as_ref()
        .filter(|cached| cached.fetched_at.elapsed() < ttl)
        .map(|cached| cached.models.clone())
}

/// 出口归一化：Claude 系模型 ID 统一为连字符（官方）写法，并去除同一模型的重复条目。
///
/// Kiro `ListAvailableModels` 返回 `claude-opus-5.5` 这类点号 ID，开启
/// `modelsOfficialAliases` 后会追加官方连字符别名 `claude-opus-5-5`，导致同一个模型
/// 在 `/v1/models` 里出现两条，客户端模型选择器出现重复项。
///
/// 处理分两步：
/// 1. 把 `claude-` 开头且含 `.` 的 ID 改写为连字符写法（`claude-opus-5.5` →
///    `claude-opus-5-5`），并按规范化后的 ID 去重，保留首次出现的条目
/// 2. 若同时存在 `claude-opus-4-5` 与带日期的 `claude-opus-4-5-20251101`，丢弃带日期
///    的那条，只保留不带日期的 ID（用户侧选择器与下游网关都用短 ID 更稳定）
///
/// 边界：
/// - 非 Claude 模型（`gpt-5.6-sol` / `deepseek-3.2` / `minimax-m2.5` 等）的点号是
///   模型名的一部分，原样保留
/// - 只有带日期 ID 而没有对应短 ID 的条目（如静态表里的 `claude-3-5-sonnet-20241022`）
///   原样保留，避免删掉唯一可用条目
/// - `map_model` 对点号与连字符两种写法映射一致，收敛后客户端请求任一保留 ID
///   都命中同一个 Kiro 模型
pub(crate) fn normalize_model_ids(models: Vec<Model>) -> Vec<Model> {
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut out: Vec<Model> = Vec::with_capacity(models.len());
    for model in models {
        let id = canonical_model_id(&model.id);
        if !seen.insert(id.clone()) {
            continue;
        }
        out.push(Model { id, ..model });
    }
    drop_dated_when_short_exists(out)
}

/// 单个模型 ID 的规范形式：`claude-` 开头且含 `.` 时把点替换为连字符，其余原样返回。
pub(crate) fn canonical_model_id(id: &str) -> String {
    if id.starts_with("claude-") && id.contains('.') {
        id.replace('.', "-")
    } else {
        id.to_string()
    }
}

/// 丢弃"存在对应短 ID"的带日期条目（纯逻辑）。
///
/// 仅处理 `claude-` 开头且以 `-YYYYMMDD` 结尾的 ID；短 ID 必须在同一份列表里真实存在
/// 才会丢弃，因此不会误删只有带日期写法的条目。
pub(crate) fn drop_dated_when_short_exists(models: Vec<Model>) -> Vec<Model> {
    let ids: std::collections::HashSet<String> = models.iter().map(|m| m.id.clone()).collect();
    models
        .into_iter()
        .filter(|m| match dated_suffix_start(&m.id) {
            Some(short) => !ids.contains(short),
            None => true,
        })
        .collect()
}

/// 若 `id` 是 `claude-*-YYYYMMDD` 形式，返回去掉日期后的短 ID；否则返回 `None`。
fn dated_suffix_start(id: &str) -> Option<&str> {
    if !id.starts_with("claude-") {
        return None;
    }
    let (short, date) = id.rsplit_once('-')?;
    if date.len() == 8 && date.bytes().all(|b| b.is_ascii_digit()) && short.starts_with("claude-") {
        Some(short)
    } else {
        None
    }
}

/// 刷新结果落地（纯逻辑，无网络）：
/// - `Some(非空)` → 写缓存并返回（分支 2）
/// - `None` 且有旧缓存 → 续用旧缓存（分支 3）
/// - `None` 且无缓存 → 静态表（分支 4）
pub(crate) fn resolve_after_refresh(
    cache: &ModelCache,
    refreshed: Option<Vec<Model>>,
) -> Vec<Model> {
    if let Some(models) = refreshed {
        *cache.write() = Some(super::super::middleware::CachedModels {
            models: models.clone(),
            fetched_at: std::time::Instant::now(),
        });
        return models;
    }
    if let Some(cached) = cache.read().as_ref() {
        return cached.models.clone();
    }
    build_model_list()
}

/// 构建可用模型列表（供 get_models 和 get_model 共用）
pub(crate) fn build_model_list() -> Vec<Model> {
    vec![
        // === 旧版模型 ID（兼容旧版 Claude Code 客户端） ===
        // 这些旧 ID 在 map_model() 中会被正确映射到对应的 Kiro 模型
        Model {
            id: "claude-3-5-sonnet-20241022".to_string(),
            object: "model".to_string(),
            created: 1729555200,
            owned_by: "anthropic".to_string(),
            display_name: "Claude 3.5 Sonnet".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 8192,
        },
        Model {
            id: "claude-3-5-haiku-20241022".to_string(),
            object: "model".to_string(),
            created: 1729555200,
            owned_by: "anthropic".to_string(),
            display_name: "Claude 3.5 Haiku".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 8192,
        },
        Model {
            id: "claude-3-opus-20240229".to_string(),
            object: "model".to_string(),
            created: 1709164800,
            owned_by: "anthropic".to_string(),
            display_name: "Claude 3 Opus".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 4096,
        },
        Model {
            id: "claude-3-haiku-20240307".to_string(),
            object: "model".to_string(),
            created: 1709769600,
            owned_by: "anthropic".to_string(),
            display_name: "Claude 3 Haiku".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 4096,
        },
        Model {
            id: "claude-3-sonnet-20240229".to_string(),
            object: "model".to_string(),
            created: 1709164800,
            owned_by: "anthropic".to_string(),
            display_name: "Claude 3 Sonnet".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 4096,
        },
        // === Claude 4.x 过渡期模型 ID ===
        Model {
            id: "claude-sonnet-4".to_string(),
            object: "model".to_string(),
            created: 1747180800,
            owned_by: "anthropic".to_string(),
            display_name: "Claude Sonnet 4".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 64000,
        },
        Model {
            id: "claude-sonnet-4-20250514".to_string(),
            object: "model".to_string(),
            created: 1747180800,
            owned_by: "anthropic".to_string(),
            display_name: "Claude Sonnet 4".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 64000,
        },
        Model {
            id: "claude-opus-4-20250514".to_string(),
            object: "model".to_string(),
            created: 1747180800,
            owned_by: "anthropic".to_string(),
            display_name: "Claude Opus 4".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 64000,
        },
        // === 当前主力模型 ===
        Model {
            id: "claude-sonnet-4-5-20250929".to_string(),
            object: "model".to_string(),
            created: 1727568000,
            owned_by: "anthropic".to_string(),
            display_name: "Claude Sonnet 4.5".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 64000,
        },
        Model {
            id: "claude-sonnet-4-5-20250929-thinking".to_string(),
            object: "model".to_string(),
            created: 1727568000,
            owned_by: "anthropic".to_string(),
            display_name: "Claude Sonnet 4.5 (Thinking)".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 64000,
        },
        Model {
            id: "claude-opus-4-5-20251101".to_string(),
            object: "model".to_string(),
            created: 1730419200,
            owned_by: "anthropic".to_string(),
            display_name: "Claude Opus 4.5".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 64000,
        },
        Model {
            id: "claude-opus-4-5-20251101-thinking".to_string(),
            object: "model".to_string(),
            created: 1730419200,
            owned_by: "anthropic".to_string(),
            display_name: "Claude Opus 4.5 (Thinking)".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 64000,
        },
        Model {
            id: "claude-sonnet-4-6".to_string(),
            object: "model".to_string(),
            created: 1770314400,
            owned_by: "anthropic".to_string(),
            display_name: "Claude Sonnet 4.6".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 64000,
        },
        Model {
            id: "claude-sonnet-4-6-thinking".to_string(),
            object: "model".to_string(),
            created: 1770314400,
            owned_by: "anthropic".to_string(),
            display_name: "Claude Sonnet 4.6 (Thinking)".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 64000,
        },
        Model {
            id: "claude-sonnet-5".to_string(),
            object: "model".to_string(),
            created: 1775600000,
            owned_by: "anthropic".to_string(),
            display_name: "Claude Sonnet 5".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 64000,
        },
        Model {
            id: "claude-sonnet-5-thinking".to_string(),
            object: "model".to_string(),
            created: 1775600000,
            owned_by: "anthropic".to_string(),
            display_name: "Claude Sonnet 5 (Thinking)".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 64000,
        },
        Model {
            id: "claude-opus-4-6".to_string(),
            object: "model".to_string(),
            created: 1770314400,
            owned_by: "anthropic".to_string(),
            display_name: "Claude Opus 4.6".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 128000,
        },
        Model {
            id: "claude-opus-4-6-thinking".to_string(),
            object: "model".to_string(),
            created: 1770314400,
            owned_by: "anthropic".to_string(),
            display_name: "Claude Opus 4.6 (Thinking)".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 128000,
        },
        Model {
            id: "claude-opus-4-7".to_string(),
            object: "model".to_string(),
            created: 1773000000,
            owned_by: "anthropic".to_string(),
            display_name: "Claude Opus 4.7".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 128000,
        },
        Model {
            id: "claude-opus-4-7-thinking".to_string(),
            object: "model".to_string(),
            created: 1773000000,
            owned_by: "anthropic".to_string(),
            display_name: "Claude Opus 4.7 (Thinking)".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 128000,
        },
        Model {
            id: "claude-opus-4-8".to_string(),
            object: "model".to_string(),
            created: 1775600000,
            owned_by: "anthropic".to_string(),
            display_name: "Claude Opus 4.8".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 128000,
        },
        Model {
            id: "claude-opus-4-8-thinking".to_string(),
            object: "model".to_string(),
            created: 1775600000,
            owned_by: "anthropic".to_string(),
            display_name: "Claude Opus 4.8 (Thinking)".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 128000,
        },
        Model {
            id: "claude-opus-5".to_string(),
            object: "model".to_string(),
            created: 1777500000,
            owned_by: "anthropic".to_string(),
            display_name: "Claude Opus 5".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 128000,
        },
        Model {
            id: "claude-opus-5-thinking".to_string(),
            object: "model".to_string(),
            created: 1777500000,
            owned_by: "anthropic".to_string(),
            display_name: "Claude Opus 5 (Thinking)".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 128000,
        },
        Model {
            id: "claude-fable-5".to_string(),
            object: "model".to_string(),
            created: 1772582400,
            owned_by: "anthropic".to_string(),
            display_name: "Claude Fable 5".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 128000,
        },
        Model {
            id: "claude-fable-5-thinking".to_string(),
            object: "model".to_string(),
            created: 1772582400,
            owned_by: "anthropic".to_string(),
            display_name: "Claude Fable 5 (Thinking)".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 128000,
        },
        Model {
            id: "claude-haiku-4-5-20251001".to_string(),
            object: "model".to_string(),
            created: 1727740800,
            owned_by: "anthropic".to_string(),
            display_name: "Claude Haiku 4.5".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 64000,
        },
        Model {
            id: "claude-haiku-4-5-20251001-thinking".to_string(),
            object: "model".to_string(),
            created: 1727740800,
            owned_by: "anthropic".to_string(),
            display_name: "Claude Haiku 4.5 (Thinking)".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 64000,
        },
        // === 非 Claude 模型 ===
        Model {
            id: "auto".to_string(),
            object: "model".to_string(),
            created: 1770314400,
            owned_by: "kiro".to_string(),
            display_name: "Auto (智能路由)".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 32000,
        },
        Model {
            id: "deepseek-3.2".to_string(),
            object: "model".to_string(),
            created: 1770314400,
            owned_by: "deepseek".to_string(),
            display_name: "DeepSeek 3.2".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 32000,
        },
        Model {
            id: "glm-5".to_string(),
            object: "model".to_string(),
            created: 1770314400,
            owned_by: "glm".to_string(),
            display_name: "GLM-5".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 32000,
        },
        Model {
            id: "minimax-m2.5".to_string(),
            object: "model".to_string(),
            created: 1770314400,
            owned_by: "minimax".to_string(),
            display_name: "MiniMax M2.5".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 32000,
        },
        Model {
            id: "minimax-m2.1".to_string(),
            object: "model".to_string(),
            created: 1770314400,
            owned_by: "minimax".to_string(),
            display_name: "MiniMax M2.1".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 32000,
        },
        Model {
            id: "qwen3-coder-next".to_string(),
            object: "model".to_string(),
            created: 1770314400,
            owned_by: "qwen".to_string(),
            display_name: "Qwen3 Coder Next".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 32000,
        },
        Model {
            id: "kimi-k3".to_string(),
            object: "model".to_string(),
            created: 1770314400,
            owned_by: "kimi".to_string(),
            display_name: "Kimi K3".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 32000,
        },
        Model {
            id: "gpt-5.6-sol".to_string(),
            object: "model".to_string(),
            created: 1770314400,
            owned_by: "openai".to_string(),
            display_name: "GPT-5.6 Sol".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 32000,
        },
        Model {
            id: "gpt-5.6-terra".to_string(),
            object: "model".to_string(),
            created: 1770314400,
            owned_by: "openai".to_string(),
            display_name: "GPT-5.6 Terra".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 32000,
        },
        Model {
            id: "gpt-5.6-luna".to_string(),
            object: "model".to_string(),
            created: 1770314400,
            owned_by: "openai".to_string(),
            display_name: "GPT-5.6 Luna".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 32000,
        },
    ]
}

/// 根据模型 ID 前缀推断提供方（ListAvailableModels 响应不含厂商归属字段）
///
/// 命名规则与 `build_model_list()` 中手工维护的 `owned_by` 保持一致；未知前缀返回 `"unknown"`。
/// 供 `/v1/models` 动态映射与 Admin 端共享，避免两份实现漂移。
pub(crate) fn guess_owned_by(model_id: &str) -> &'static str {
    let id = model_id.to_lowercase();
    if id.contains("claude") {
        "anthropic"
    } else if id.contains("gpt") {
        "openai"
    } else if id == "auto" {
        "kiro"
    } else if id.contains("deepseek") {
        "deepseek"
    } else if id.contains("minimax") {
        "minimax"
    } else if id.contains("glm") {
        "glm"
    } else if id.contains("qwen") {
        "qwen"
    } else if id.contains("kimi") {
        "kimi"
    } else {
        "unknown"
    }
}

/// 将上游 `ListAvailableModels` 返回的单条模型映射为 Anthropic `Model`
///
/// 纯函数，不涉及网络调用，可直接用 fake `AvailableModelInfo` 单测。
pub(crate) fn available_model_to_model(
    info: &crate::kiro::model::available_models::AvailableModelInfo,
) -> Model {
    Model {
        id: info.model_id.clone(),
        object: "model".to_string(),
        created: 0,
        owned_by: guess_owned_by(&info.model_id).to_string(),
        display_name: info.model_name.clone(),
        model_type: "chat".to_string(),
        max_tokens: info.token_limits.max_output_tokens as i32,
    }
}

/// GET /v1/models/:model_id
///
/// 返回指定模型的信息
pub async fn get_model(
    State(state): State<AppState>,
    axum::extract::Path(model_id): axum::extract::Path<String>,
) -> Response {
    tracing::info!(model_id = %model_id, "Received GET /v1/models/:model_id request");

    // 与 /v1/models 相同的动态来源，查找匹配的模型
    let models = fetch_models_dynamic(&state).await;
    if let Some(model) = models.into_iter().find(|m| m.id == model_id) {
        Json(model).into_response()
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse::new(
                "not_found_error",
                format!("Model '{}' not found", model_id),
            )),
        )
            .into_response()
    }
}

#[cfg(test)]
mod alias_tests {
    use super::*;

    fn m(id: &str) -> Model {
        Model {
            id: id.to_string(),
            object: "model".to_string(),
            created: 0,
            owned_by: guess_owned_by(id).to_string(),
            display_name: id.to_string(),
            model_type: "chat".to_string(),
            max_tokens: 128000,
        }
    }

    fn ids(models: &[Model]) -> Vec<&str> {
        models.iter().map(|m| m.id.as_str()).collect()
    }

    #[test]
    fn adds_hyphen_alias_after_dotted_claude_id() {
        let out = with_official_aliases(vec![m("claude-opus-5.5"), m("claude-opus-5"), m("auto")]);
        assert_eq!(
            ids(&out),
            [
                "claude-opus-5.5",
                "claude-opus-5-5",
                "claude-opus-5",
                "auto"
            ]
        );
        assert_eq!(out[1].max_tokens, out[0].max_tokens);
        assert_eq!(out[1].display_name, "claude-opus-5.5");
    }

    #[test]
    fn non_claude_dotted_ids_untouched() {
        let out =
            with_official_aliases(vec![m("gpt-5.6-sol"), m("deepseek-3.2"), m("minimax-m2.5")]);
        assert_eq!(ids(&out), ["gpt-5.6-sol", "deepseek-3.2", "minimax-m2.5"]);
    }

    #[test]
    fn four_five_gets_dated_official_id() {
        let out = with_official_aliases(vec![m("claude-haiku-4.5")]);
        assert_eq!(
            ids(&out),
            [
                "claude-haiku-4.5",
                "claude-haiku-4-5",
                "claude-haiku-4-5-20251001"
            ]
        );
    }

    #[test]
    fn existing_alias_not_duplicated() {
        let out = with_official_aliases(vec![m("claude-opus-4.6"), m("claude-opus-4-6")]);
        assert_eq!(ids(&out), ["claude-opus-4.6", "claude-opus-4-6"]);
    }

    /// 别名必须与原 ID 映射到同一个 Kiro 模型，否则暴露出去的别名请求会打到别的模型
    #[test]
    fn every_alias_maps_to_same_kiro_model() {
        let upstream = [
            "claude-opus-5.5",
            "claude-sonnet-5.5",
            "claude-opus-4.8",
            "claude-opus-4.7",
            "claude-opus-4.6",
            "claude-sonnet-4.6",
            "claude-opus-4.5",
            "claude-sonnet-4.5",
            "claude-haiku-4.5",
        ];
        let out = with_official_aliases(upstream.iter().map(|id| m(id)).collect());
        for model in &out {
            let base = upstream
                .iter()
                .find(|u| {
                    model.id == **u
                        || model.id == u.replace('.', "-")
                        || DATED_OFFICIAL_IDS
                            .iter()
                            .any(|(d, a)| d == *u && *a == model.id)
                })
                .unwrap_or_else(|| panic!("unexpected id {}", model.id));
            assert_eq!(
                crate::anthropic::converter::map_model(&model.id),
                crate::anthropic::converter::map_model(base),
                "{} 与 {} 映射不一致",
                model.id,
                base
            );
        }
        assert_eq!(out.len(), upstream.len() * 2 + 3);
    }

    // ==== 出口归一化（normalize_model_ids / canonical_model_id）====

    /// 核心诉求：点号 ID 与连字符 ID 同时存在时，出口只保留连字符那条
    #[test]
    fn normalize_drops_dotted_keeps_hyphen() {
        let out = normalize_model_ids(vec![m("claude-opus-5.5"), m("claude-opus-5-5")]);
        assert_eq!(ids(&out), ["claude-opus-5-5"]);
    }

    /// 只有点号 ID（未开别名开关）时，出口把它改写成连字符写法
    #[test]
    fn normalize_rewrites_lone_dotted_id() {
        let out = normalize_model_ids(vec![m("claude-opus-4.6"), m("auto")]);
        assert_eq!(ids(&out), ["claude-opus-4-6", "auto"]);
        assert_eq!(out[0].max_tokens, 128000, "改写 ID 不应丢字段");
    }

    /// 4.5 代际：短 ID 与带日期 ID 并存时，只保留不带日期的短 ID
    #[test]
    fn normalize_prefers_short_id_over_dated() {
        let out = normalize_model_ids(vec![
            m("claude-haiku-4.5"),
            m("claude-haiku-4-5"),
            m("claude-haiku-4-5-20251001"),
        ]);
        assert_eq!(ids(&out), ["claude-haiku-4-5"]);
    }

    /// 只有带日期 ID 时不能删（否则该模型彻底消失）
    #[test]
    fn normalize_keeps_dated_id_when_short_absent() {
        let out = normalize_model_ids(vec![m("claude-opus-4-5-20251101")]);
        assert_eq!(ids(&out), ["claude-opus-4-5-20251101"]);
    }

    /// 非 Claude 模型点号属于名字本身，一律不动
    #[test]
    fn normalize_leaves_non_claude_dots() {
        let out = normalize_model_ids(vec![
            m("gpt-5.6-sol"),
            m("deepseek-3.2"),
            m("minimax-m2.5"),
            m("glm-5"),
        ]);
        assert_eq!(
            ids(&out),
            ["gpt-5.6-sol", "deepseek-3.2", "minimax-m2.5", "glm-5"]
        );
    }

    /// 去重后不能残留任何重复 ID
    #[test]
    fn normalize_output_has_no_duplicate_ids() {
        let input = vec![
            m("claude-opus-5.5"),
            m("claude-opus-5-5"),
            m("claude-opus-4.8"),
            m("claude-opus-4-8"),
            m("claude-opus-5"),
            m("auto"),
        ];
        let out = normalize_model_ids(input);
        let uniq: std::collections::HashSet<&str> = ids(&out).into_iter().collect();
        assert_eq!(uniq.len(), out.len(), "出口存在重复 ID");
        assert_eq!(ids(&out), ["claude-opus-5-5", "claude-opus-4-8", "claude-opus-5", "auto"]);
    }

    /// 静态表经过出口归一化后同样不应出现重复
    #[test]
    fn normalize_static_list_has_no_duplicates() {
        let out = normalize_model_ids(build_model_list());
        let uniq: std::collections::HashSet<&str> = ids(&out).into_iter().collect();
        assert_eq!(uniq.len(), out.len(), "静态表出口存在重复 ID");
        // 静态表里的带日期条目没有对应短 ID 时必须保留
        assert!(ids(&out).contains(&"claude-opus-4-5-20251101"));
        assert!(ids(&out).contains(&"claude-sonnet-4-5-20250929"));
        assert!(ids(&out).contains(&"claude-haiku-4-5-20251001"));
    }

    /// 归一化前后 map_model 结果一致（否则收敛会改变请求落点）
    #[test]
    fn normalize_preserves_map_model_target() {
        for id in [
            "claude-opus-5.5",
            "claude-sonnet-5.5",
            "claude-opus-4.8",
            "claude-opus-4.7",
            "claude-opus-4.6",
            "claude-sonnet-4.6",
        ] {
            let canonical = canonical_model_id(id);
            assert_eq!(
                crate::anthropic::converter::map_model(id),
                crate::anthropic::converter::map_model(&canonical),
                "{id} 与 {canonical} 映射不一致"
            );
        }
    }

    #[test]
    fn canonical_model_id_cases() {
        assert_eq!(canonical_model_id("claude-opus-5.5"), "claude-opus-5-5");
        assert_eq!(canonical_model_id("claude-opus-5-5"), "claude-opus-5-5");
        assert_eq!(canonical_model_id("gpt-5.6-sol"), "gpt-5.6-sol");
        assert_eq!(canonical_model_id("deepseek-3.2"), "deepseek-3.2");
        assert_eq!(canonical_model_id("auto"), "auto");
    }

    /// 端到端：开启别名后（点号 + 连字符 + 带日期三份）出口只留一条
    #[test]
    fn aliases_then_normalize_yields_single_entry() {
        let aliased = with_official_aliases(vec![
            m("claude-opus-4.5"),
            m("claude-opus-5.5"),
            m("gpt-5.6-sol"),
        ]);
        let out = normalize_model_ids(aliased);
        assert_eq!(ids(&out), ["claude-opus-4-5", "claude-opus-5-5", "gpt-5.6-sol"]);
    }
}
