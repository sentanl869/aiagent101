from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import logging
import os
import uuid
import datetime
from typing import Annotated, TypedDict
from functools import partial

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("requirement_engine.log", encoding='utf-8') 
    ]
)
logger = logging.getLogger("RequirementEngine")

# 从环境变量获取API配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")

# 初始化LLM
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    model="deepseek-ai/DeepSeek-R1",
    temperature=0.3
)

class State(TypedDict):
    requirements: Annotated[str, "原始需求描述"]
    analysis: Annotated[str, "需求分解结果"]
    document: Annotated[str, "需求文档草稿"]
    final_doc: Annotated[str, "最终需求文档"]
    file_path: Annotated[str, "生成的Markdown文件路径"]  # 新增文件路径字段[7](@ref)

# 专业prompt模板
REQUIREMENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "作为资深产品经理，请对以下需求进行结构化分析："),
    ("human", """需求描述: {requirements}

输出格式要求：
1. 用户故事（包含角色、场景、价值）
2. 功能模块分解
3. 非功能性需求
4. 风险点评估""")
])

DOCUMENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "请根据以下分析结果生成标准需求文档："),
    ("human", """分析结果: {analysis}

文档结构要求：
1. 项目背景
2. 目标范围
3. 功能详述
4. 系统架构图（文字描述）
5. 验收标准""")
])

# 优化后的处理节点
def analyze_requirements(state: State) -> dict:
    """需求分析节点（带摘要日志）"""
    logger.info(f"开始需求分析: {state['requirements'][:50]}...")
    
    # 使用LCEL管道提高效率
    chain = REQUIREMENT_PROMPT | llm
    response = chain.invoke({"requirements": state["requirements"]})
    
    # 摘要日志代替完整输出
    analysis_summary = response.content[:100] + "..." if len(response.content) > 100 else response.content
    logger.info(f"需求分析完成，摘要: {analysis_summary}")
    
    return {"analysis": response.content}

def generate_document(state: State) -> dict:
    """文档生成节点（带摘要日志）"""
    logger.info(f"开始生成文档，分析长度: {len(state['analysis'])}字符")
    
    # 使用LCEL管道提高效率
    chain = DOCUMENT_PROMPT | llm
    response = chain.invoke({"analysis": state["analysis"]})
    
    # 摘要日志代替完整输出
    doc_summary = response.content[:100] + "..." if len(response.content) > 100 else response.content
    logger.info(f"文档生成完成，摘要: {doc_summary}")
    
    return {"document": response.content}

def finalize_output(state: State) -> dict:
    """最终输出节点"""
    logger.info("生成最终需求文档")
    return {"final_doc": f"# 需求文档\n\n{state['document']}"}  # 添加Markdown标题

def save_to_markdown(state: State) -> dict:
    """将最终文档保存为Markdown文件并返回路径"""
    # 创建输出目录（如果不存在）
    output_dir = "output_docs"
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成唯一文件名（基于时间戳和UUID）
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    filename = f"requirement_doc_{timestamp}_{unique_id}.md"
    file_path = os.path.join(output_dir, filename)
    
    # 写入Markdown文件
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(state["final_doc"])
    
    logger.info(f"文档已保存至: {file_path}")
    return {"file_path": file_path}

# 工作流构建
graph_builder = StateGraph(State)

# 添加节点
graph_builder.add_node("analyze", analyze_requirements)
graph_builder.add_node("generate", generate_document)
graph_builder.add_node("finalize", finalize_output)
graph_builder.add_node("save", save_to_markdown)  # 新增保存节点

# 构建工作流
graph_builder.add_edge(START, "analyze")
graph_builder.add_edge("analyze", "generate")
graph_builder.add_edge("generate", "finalize")
graph_builder.add_edge("finalize", "save")  # 最终输出后保存
graph_builder.add_edge("save", END)  # 保存后结束工作流

# 编译工作流
graph = graph_builder.compile()

# 使用示例
def generate_requirement_doc(requirement_str: str) -> str:
    """
    生成需求文档的对外接口函数
    
    参数:
        requirement_str (str): 需求描述字符串
        
    返回:
        str: 生成的需求文档文件路径
    """
    try:
        logger.info(f"开始处理需求: {requirement_str[:50]}...")
        result = graph.invoke({
            "requirements": requirement_str,
            "analysis": "",
            "document": "",
            "final_doc": "",
            "file_path": ""
        })
        return result["file_path"]
    except Exception as e:
        logger.error(f"文档生成失败: {str(e)}", exc_info=True)
        raise RuntimeError(f"需求文档生成失败: {str(e)}") from e

# 添加到文件末尾，确保在graph定义之后
if __name__ == "__main__":
    # 测试用例
    test_path = generate_requirement_doc("测试需求：用户需要课程评价功能")
    print(f"生成文档路径: {test_path}")
    