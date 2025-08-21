import requests
import json

# BASE_URL = "https://freetier-01.cn-hangzhou.cluster.cn-dev.matrixone.tech"
# BASE_URL = "https://moi.cn-dev.matrixorigin.cn"
BASE_URL = "https://freetier-01.cn-hangzhou.cluster.matrixonecloud.cn"

username = "admin"
password = "123ZhengZhiYue"
account_name = "019855bb-ee9f-7459-8658-b5302c82aee2"

catalog_name = "默认"
database_name = "target"
target_volume_name = "moi-helper2"
branch_name = "主要"

headers = {
    "user-id": "0197f384-7639-78c0-a4eb-5941d2184b2d",
    "Access-Token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOiIwMTkyYmUwMy01OTkyLTc2NmQtYjI5Zi1iNzgzNTQ3ZjkyMzQiLCJuYW1lIjoi5L2z5L-KIOaWuSIsImVtYWlsIjoiNjUzNjM5NzkxQHFxLmNvbSIsImxvZ2luX21ldGhvZCI6ImJhc2ljIiwiZXhwIjoxNzU1NzQ1OTAwLCJpYXQiOjE3NTU3NDUwMDAsImlzcyI6Im1vaSJ9.nb2iC_okhyitAwTLoP3QxSCStToXRZ6igmxz86ij0DxYSBN0Oi6X3H3LXgcF_TGc2Jho8in3ta3Nlq9g3dmoHsj79Xq1LQRArldBvHBgpVpItKJJ8Y0O1juZnPBgvXS_6XE28v5aXM8UPMo9v70qNG1qjmUQz5BU1WSUWrh3zYM",
    "uid": "0192be03-5992-766d-b29f-b783547f9234-0197f384-7639-78c0-a4eb-5941d2184b2d:admin:accountadmin"
}

# username = "admin"
# password = "Admin123"
# account_name = "0197f384-7639-78c0-a4eb-5941d2184b2d"

catalog_name = "默认"
database_name = "目标数据卷"
target_volume_name = "兆易创新0821"
branch_name = "主要"


def login() -> dict:
    login_url = BASE_URL + "/auth/login"
    login_headers = {
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/json"
    }   
    login_body = {
        "account_name": account_name,
        "username": username,
        "password": password,
        "type":"workspace",
    }
    try:
        login_response = requests.post(login_url, headers=login_headers, json=login_body, timeout=15)
        login_response.raise_for_status()

        # 从响应头获取 Access-Token
        access_token = login_response.headers.get("Access-Token", "")
        
        # 从响应体中获取 uid
        try:
            login_data = login_response.json()
            uid = login_data.get("data", {}).get("uid", "")

            headers = {
                "user-id": account_name,
                "Access-Token": access_token,
                "uid": uid
            }
            print(headers)
            print("===== 登录成功 =====")
            # print("Access-Token:", access_token)
            # print("UID:", uid)
            # print()
            return headers
        except json.JSONDecodeError:
            uid = ""
            print("解析登录响应 JSON 出错，请检查返回格式。")

    except requests.exceptions.RequestException as e:
        print("登录请求失败:", e)
        raise SystemExit

def get_catalog_id(headers: dict) -> str:
    try:
        catalog_list = requests.post(f"{BASE_URL}/catalog/list", headers=headers)
        print(catalog_list.json())
    except requests.exceptions.RequestException as e:
        print("获取 catalog 列表失败:", e)
        raise SystemExit
    
    if catalog_list.status_code == 200:
        catalog_list_data = catalog_list.json()
        if "data" not in catalog_list_data or "list" not in catalog_list_data.get("data", {}):
            print("获取 catalog 列表返回异常:", catalog_list_data)
            raise SystemExit("鉴权失败或接口返回异常（catalog/list）。")
        for catalog in catalog_list_data["data"]["list"]:
            if catalog["name"] == catalog_name:
                return catalog["id"]
    return ""

def get_database_id(headers: dict, catalog_id: str) -> str:
    try:
        database_list = requests.post(f"{BASE_URL}/catalog/database/list", headers=headers, json={"id": catalog_id})
    except requests.exceptions.RequestException as e:
        print("获取 database 列表失败:", e)
        raise SystemExit
    
    if database_list.status_code == 200:
        database_list_data = database_list.json()
        if "data" not in database_list_data or "list" not in database_list_data.get("data", {}):
            print("获取 database 列表返回异常:", database_list_data)
            raise SystemExit("鉴权失败或接口返回异常（catalog/database/list）。")
        for database in database_list_data["data"]["list"]:
            if database["name"] == database_name:
                return database["id"]
    return ""

def get_volume_id(headers: dict, database_id: str) -> str:
    try:
        volume_list = requests.post(f"{BASE_URL}/catalog/database/children", headers=headers, json={"id": database_id})
    except requests.exceptions.RequestException as e:
        print("获取 volume 列表失败:", e)
        raise SystemExit
    
    if volume_list.status_code == 200:
        volume_list_data = volume_list.json()
        if "data" not in volume_list_data or "list" not in volume_list_data.get("data", {}):
            print("获取 volume 列表返回异常:", volume_list_data)
            raise SystemExit("鉴权失败或接口返回异常（catalog/database/children）。")
        for volume in volume_list_data["data"]["list"]:
            if volume["name"] == target_volume_name:
                return volume["id"]
    return ""

def get_branch_id(headers: dict, volume_id: str) -> str:
    try:
        body = {
            "filters": [
                {
                    "name": "volume_id",
                    "values": [
                        volume_id
                    ]
                }
            ]
        }
        branch_list = requests.post(f"{BASE_URL}/catalog/file/list", headers=headers, json=body)
    except requests.exceptions.RequestException as e:
        print("获取 branch 列表失败:", e)
        raise SystemExit
    
    if branch_list.status_code == 200:
        branch_list_data = branch_list.json()
        if "data" not in branch_list_data or "list" not in branch_list_data.get("data", {}):
            print("获取 branch 列表返回异常:", branch_list_data)
            raise SystemExit("鉴权失败或接口返回异常（catalog/file/list 查询 branch）。")
        for branch in branch_list_data["data"]["list"]:
            if branch["name"] == branch_name:
                return branch["id"]
    return ""

def get_file_id(headers: dict, branch_id: str) -> list:
    file_ids = []
    page = 1
    page_size = 100
    
    while True:
        try:
            body = {
                "filters": [
                    {
                        "name": "parent_id",
                        "values": [
                            branch_id
                        ]
                    }
                ],
                "page": page,
                "page_size": page_size
            }
            file_list = requests.post(f"{BASE_URL}/catalog/file/list", headers=headers, json=body)
        except requests.exceptions.RequestException as e:
            print("获取 file 列表失败:", e)
            raise SystemExit
        
        if file_list.status_code == 200:
            file_list_data = file_list.json()
            if "data" not in file_list_data or "list" not in file_list_data.get("data", {}):
                print("获取 file 列表返回异常:", file_list_data)
                break
            current_files = file_list_data["data"]["list"]
            
            # 将当前页的文件ID添加到列表中
            for file in current_files:
                file_ids.append(file["ref_file_id"])
            
            print(f"已获取第 {page} 页，共 {len(current_files)} 个文件")
            
            # 如果当前页的文件数量小于页大小，说明已经是最后一页了
            if len(current_files) < page_size:
                break
            
            page += 1
        else:
            print(f"请求失败，状态码: {file_list.status_code}")
            break
    
    print(f"总共获取到 {len(file_ids)} 个文件ID")
    return file_ids

def get_file_segments(headers: dict, branch_id: str, file_id: str) -> list:
    offset = 0
    limit = 10
    total = None
    blocks = []
    content_types = []
    while True:
        try:
            body = {
                "offset": offset,
                "limit": limit
            }
            file_segments = requests.post(f"{BASE_URL}/byoa/api/v1/explore/volumes/{branch_id}/files/{file_id}/blocks", headers=headers, json=body)
            # print(file_segments.json())
        except requests.exceptions.RequestException as e:
            print("获取 file 分段失败:", e)
            raise SystemExit
        
        if file_segments.status_code == 200:
            resp_json = file_segments.json()
            if "data" not in resp_json:
                print("获取 file 分段返回异常:", resp_json)
                break
            file_segments_data = resp_json["data"]
            if not total:
                total = file_segments_data["total"]
                print(f"文件 {file_id} 总共有 {total} 个分段")
            for item in file_segments_data["items"]:
                content_types.append(item["content_type"])
                blocks.append(item["content"])
            if len(blocks) >= total or len(file_segments_data["items"]) < limit:
                break
            offset += limit
        else:
            print(f"请求失败，状态码: {file_segments.status_code}")
            break
    return blocks, content_types

def save_all_segments_to_json(output_file: str = "data/file_segments.json"):
    """
    获取所有文档段落并保存为JSON格式
    
    Args:
        output_file: 输出文件路径
    """
    import os
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 登录并获取必要的ID
    # headers = login()
    catalog_id = get_catalog_id(headers)
    print(f"catalog_id: {catalog_id}")
    database_id = get_database_id(headers, catalog_id)
    print(f"database_id: {database_id}")
    volume_id = get_volume_id(headers, database_id)
    print(f"volume_id: {volume_id}")
    branch_id = get_branch_id(headers, volume_id)
    print(f"branch_id: {branch_id}")
    
    # 获取所有文件ID
    file_ids = get_file_id(headers, branch_id)
    
    # 收集所有段落数据
    all_segments = []
    total_segments = 0
    
    print(f"开始处理 {len(file_ids)} 个文件...")
    
    for i, file_id in enumerate(file_ids):
        # try:
        print(f"处理文件 {i+1}/{len(file_ids)}: {file_id}")
        blocks, content_types = get_file_segments(headers, branch_id, file_id)
        
        # 为每个段落创建结构化数据
        for j, block_content in enumerate(blocks):
            # print(block_content)
            segment_data = {
                "segment_id": f"{file_id}_{j}",
                "file_id": file_id,
                "block_index": j,
                "content": block_content.strip(),
                "content_type": content_types[j],
                "content_length": len(block_content.strip()),
                "word_count": len(block_content.strip().split())
            }
            
            # 过滤掉过短的段落
            if segment_data["content_length"] > 10:
                all_segments.append(segment_data)
                total_segments += 1
        
        print(f"文件 {file_id}: 提取 {len(blocks)} 个段落")
            
        # except Exception as e:
        #     print(f"处理文件 {file_id} 时出错: {e}")
        #     continue
    
    # 保存到JSON文件
    final_data = {
        "metadata": {
            "total_files": len(file_ids),
            "total_segments": len(all_segments),
            "average_content_length": sum(s["content_length"] for s in all_segments) / len(all_segments) if all_segments else 0,
            "catalog_name": catalog_name,
            "database_name": database_name,
            "volume_name": target_volume_name,
            "branch_name": branch_name,
            "extracted_at": json.dumps({"timestamp": "now"})  # 会在保存时自动填充
        },
        "segments": all_segments
    }
    
    # 添加时间戳
    from datetime import datetime
    final_data["metadata"]["extracted_at"] = datetime.now().isoformat()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    
    print(f"===== 数据保存完成 =====")
    print(f"输出文件: {output_file}")
    print(f"总文档数: {len(file_ids)}")
    print(f"总段落数: {len(all_segments)}")
    print(f"平均段落长度: {final_data['metadata']['average_content_length']:.2f} 字符")
    
    return output_file

if __name__ == "__main__":
    # 保存完整的段落数据用于训练
    output_file = save_all_segments_to_json()
    print(f"文档段落数据已保存到: {output_file}")