import asyncio
import httpx
import pandas as pd
import traceback    
import os

current_dir = os.getcwd()
data_path = os.path.join(current_dir, 'data')

# Load your test CSV
df = pd.read_csv(os.path.join(data_path, "test.csv")) 

async def send_request(client, row_dict):
    try:
        response = await client.post("http://127.0.0.1:8000/predict", json=row_dict)
        print(response.json())
    except Exception as e:
        print("Error:", e)
        traceback.print_exc()

async def main():
    async with httpx.AsyncClient() as client:
        tasks = []
        for _ in range(100):  # simulate 100 users
            row = df.sample(1).iloc[0]
            row_dict = row.to_dict()
            tasks.append(send_request(client, row_dict))
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
