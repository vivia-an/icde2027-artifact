"""SDCCheck - 分布式训练数据一致性检查工具"""

# 核心基础模块
from .schema_analyzer import SchemaAnalyzer
from .database_executor import DatabaseExecutor

# LLM Agent实现
from .llm_orchestrator import LLMOrchestrator
from .generators import (
    LLMConstraintGenerator,
    LLMSQLGenerator,
    LLMResultAnalyzer
)
from .agents import (
    BaseAgent,
    ConstraintAgent,
    SQLAgent,
    AnalysisAgent
)
from .config import LLMConfig, PREDEFINED_CONFIGS, get_config
from .llm_providers import (
    LLMProvider,
    LLMProviderFactory,
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
    MockProvider
)

# Token 追踪
from .token_tracker import (
    TokenTracker,
    SQLTokenConsumptionManager,
    get_token_tracker,
    get_consumption_manager
)

# 数据模型
from .models import (
    TrainingConfig,
    SchemaInfo,
    Constraint,
    ConstraintType,
    AnalysisReport,
    ResultStatus
)

__version__ = "1.0.0"

__all__ = [
    # 核心基础模块
    "SchemaAnalyzer",
    "DatabaseExecutor",
    
    # LLM Agent实现
    "LLMOrchestrator",
    "LLMConstraintGenerator",
    "LLMSQLGenerator",
    "LLMResultAnalyzer",
    "LLMConfig",
    "BaseAgent",
    "ConstraintAgent",
    "SQLAgent",
    "AnalysisAgent",
    "LLMProvider",
    "LLMProviderFactory",
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "MockProvider",
    "PREDEFINED_CONFIGS",
    "get_config",
    
    # Token 追踪
    "TokenTracker",
    "SQLTokenConsumptionManager",
    "get_token_tracker",
    "get_consumption_manager",
    
    # 数据模型
    "TrainingConfig",
    "SchemaInfo",
    "Constraint",
    "ConstraintType",
    "AnalysisReport",
    "ResultStatus"
]