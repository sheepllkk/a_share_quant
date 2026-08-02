import requests

try:
    # 请求一个能返回你当前真实 IP 和地理位置的接口
    res = requests.get("http://ip-api.com/json/", timeout=5)
    data = res.json()
    print(f"🌍 Python 当前网络位置: {data.get('country')} - {data.get('city')}")
    print(f"💻 当前使用的 IP: {data.get('query')}")
except Exception as e:
    print(f"❌ 网络请求完全堵塞: {e}")