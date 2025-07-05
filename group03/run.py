from agents import generate_requirement_doc

if __name__ == "__main__":
    doc_path = generate_requirement_doc("实现一个用户注册登录页面")
    print(f"需求文档已生成，路径: {doc_path}")
