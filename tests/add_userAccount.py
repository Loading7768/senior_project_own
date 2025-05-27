import asyncio

from twikit import Client, TooManyRequests
import json
import httpx

from random import randint, uniform
import random
from datetime import datetime

import os
from tqdm import tqdm
import re


'''可修改參數'''
FOLDER_PATH = "../temp_data/02"  # 選擇要對哪個資料夾執行

USER_ACCOUNT_PATH = "./user_account.json"  # 設定已儲存的 user_account 在哪裡

JSON_DICT_NAME = "dogecoin"  # 設定推文所存的 json 檔中字典的名稱

SEARCH = 'Latest'  # 在 X 的哪個欄位內搜尋 (Top, Latest, People, Media, Lists)
'''可修改參數'''





# 定義 **異步** 函式來獲取推文
async def get_tweets(client, tweets, query):
    # if tweets is None:

    # 獲取推文
    # print(f'{datetime.now()} - Getting tweets...')
    tweets = await client.search_tweet(query, product=SEARCH)

    # 等待一段隨機時間後再獲取下一批推文
    base = randint(4, 7)
    wait_time = base + uniform(0.2, 1.5)  # 更自然的等待
    print(f'{datetime.now()} - Getting next tweets after {wait_time} seconds ...')
    await asyncio.sleep(wait_time)  # `await` 讓程式非同步等待
    tweets = await tweets.next()
    
    # 低機率的長時間等待 模擬人突然去忙別的事
    if random.random() < 0.05:  # 5% 機率
        break_time = randint(60, 180)  # 等 1 到 3 分鐘
        print(f'{datetime.now()} - Taking a short break for {break_time} seconds...')
        await asyncio.sleep(break_time)

    return tweets



# 定義預處理函數（移除 \n 與 http 連結）
async def preprocess(text):
    # 移除 \n
    # re.sub(…)：表示將符合這個模式的部分通通移除
    text = re.sub(r'\n', ' ', text)

    # 移除網址（http 開頭）
    # \S+：一串不是空白的字元
    text = re.sub(r'http\S+', '', text)

    return text



# 定義 **異步** 主函式
async def main():
    """主執行函式"""
    # 登入 X.com（原 Twitter）
    # 1) 直接在登入後的 X 上抓出 "auth_token", "ct0"
    # 2) 儲存並加載 Cookies 來保持登入狀態
    client = Client(language='en-US')
    client.load_cookies('cookies2.json')  # 這裡 **不用 await**，因為是同步函式

    # 設定目前是否達到此帳號抓文的上限
    TooManyRequests_bool = False

    # 先將儲存 "未成功抓到 user_account 的 tweet" 的 txt 檔清空
    txtname = f"{FOLDER_PATH}/notFoundAccount.txt"
    with open(txtname, 'w', encoding="utf-8-sig") as filetxt:
        filetxt.write("")

    # 讀取目前已經有的帳戶名 json 檔
    with open(USER_ACCOUNT_PATH, 'r', encoding='utf-8-sig') as fileuser:
        user_account_json = json.load(fileuser)


    # 讀入資料夾中的 json 檔
    all_files = [f for f in os.listdir(FOLDER_PATH) if f.endswith(".json")]
    print(f"📂 總共找到 {len(all_files)} 個檔案要處理")

    for file in all_files:
        filepath = os.path.join(FOLDER_PATH, file)
        jsonfilename = os.path.basename(filepath)  # ex: DOGE_20210428.json

        # 讀入要修改的 json 檔案
        with open(filepath, 'r', encoding='utf-8-sig') as file:
            data_json = json.load(file)

        json_tweets = data_json[JSON_DICT_NAME]
        print(f"\n📄 正在處理檔案：{jsonfilename}，共 {len(json_tweets)} 筆推文")

        for i, everyTweet in enumerate(tqdm(json_tweets, desc=f"🔍 處理 {jsonfilename}")):
            # 將每筆推文的 text 都直接當作 QUERY
            # 且把   \n -> " "     https... -> ""
            QUERY = await preprocess(everyTweet["text"])
            username = everyTweet["username"]
            print()
            print(f"username: {repr(username)[1:-1]}")
            print(repr(QUERY)[1:-1])


            # 先在 user_account.json 檔中找看看之前是否已經抓過這個帳號的 user.screen_name (帳戶名)
            if username in user_account_json:
                everyTweet["user_account"] = user_account_json[username]
                print(f"已找過此帳戶名: {user_account_json[username]}")

            # 否則利用 twikit 上網找
            else:
                # 初始化 tweets
                tweets = None

                # 將 action 都先設為有抓到推文
                action = "get"

                start_time = datetime.now()
                TooManyRequests_last = start_time
                    
                try:
                    tweets = await get_tweets(client, tweets, QUERY)  # `await` 確保非同步運行
                    action = "get"
                except TooManyRequests as e:
                    # 如果 MINIMUM_TWEETS 太大導致達到 API 限制，等待直到限制解除
                    rate_limit_reset = datetime.fromtimestamp(e.rate_limit_reset)
                    print(f'{datetime.now()} - Rate limit reached. Waiting until {rate_limit_reset}')
                    wait_time = (rate_limit_reset - datetime.now()).total_seconds()

                    
                    # 如果在 action == "limit" 表示上一個動作也是 rate limit reached => 代表此帳號今天的抓文達上限 => 立刻 break
                    # difference = datetime.now() - TooManyRequests_last
                    if action == "limit" and TooManyRequests_last != start_time:
                        TooManyRequests_bool = True
                        print(f"{datetime.now()} - This account rate limit reached - TooManyRequests twice")
                        break

                    # 紀錄本次 ToManyRequests 的時間 來當作下一次的參考時間
                    TooManyRequests_last = datetime.now()

                    action = "limit"
                    await asyncio.sleep(wait_time)  # `await` 讓程式非同步等待
                    continue
                except httpx.ConnectTimeout:
                    # 如果無法正常登入 X (發生 time out) 則等待 10 分鐘（600 秒）
                    print(f'{datetime.now()} - Connection timed out. Retrying in 10 minutes...')

                    await asyncio.sleep(600)  # `await` 讓程式非同步等待
                    continue
                except httpx.ReadTimeout: # as e 讀不出東西
                    # 表示程式在嘗試從伺服器讀取資料時超時
                    print(f'{datetime.now()} - Read timeout occurred. Retrying in a few seconds...')
                    await asyncio.sleep(randint(5, 15)) # 等待一段時間後重試

                    continue

                if not tweets:
                    # 如果找不到推文
                    with open(txtname, 'a', encoding="utf-8-sig") as filetxt:
                        filetxt.write(f"{jsonfilename}:\n"
                                    f"    tweet_count: {everyTweet["tweet_count"]}\n"
                                    f"    username: [{everyTweet["username"]}]\n"
                                    f"    text: [{repr(QUERY)[1:-1]}]\n"
                                    f"    (沒有找到推文)\n\n")
                    print(f'{datetime.now()} - 沒有找到推文')

                    continue


                # 把 user_account 加進 json 檔中
                everyTweet["user_account"] = tweets[0].user.screen_name  # 使用者帳戶名
                print(f"user_account: {everyTweet["user_account"]}")

                if len(tweets) > 1:
                    # 如果抓到的推文不唯一
                    with open(txtname, 'a', encoding="utf-8-sig") as filetxt:
                        filetxt.write(f"{jsonfilename}:\n"
                                    f"    tweet_count: {everyTweet["tweet_count"]}\n"
                                    f"    username: [{everyTweet["username"]}]\n"
                                    f"    text: [{repr(QUERY)[1:-1]}]\n"
                                    f"    (推文數量不唯一)\n\n")
                    print(f'{datetime.now()} - 推文數量不唯一')

                # 把 user_account 加進 user_account.json 檔中
                user_account_json[username] = everyTweet["user_account"]

                # 讀取目前已經有的帳戶名 json 檔
                with open(USER_ACCOUNT_PATH, 'w', encoding='utf-8-sig') as fileuser:
                    json.dump(user_account_json, fileuser, indent=4, ensure_ascii=False)
            
            # 每 5 筆就寫回原本推文的 json 檔中
            if (i + 1) % 5 == 0:
                print("已執行 5 次 寫回 json 檔")
                with open(filepath, 'w', encoding='utf-8-sig') as f_json:  
                    json.dump(data_json, f_json, indent=4, ensure_ascii=False)

        # 等一個 json 檔全部跑完後再寫回原本的檔案中
        with open(filepath, 'w', encoding='utf-8-sig') as f_json:  
            json.dump(data_json, f_json, indent=4, ensure_ascii=False)

        # 爬取結束
        print(f'{datetime.now()} - ✅ user_account is added')
        

        if TooManyRequests_bool:
            break
        
        base = randint(4, 7)  # 基礎時間 4s ~ 7s
        wait_time_last = base + uniform(0.2, 1.5)  # 加一點小的隨機毫秒
        wait_time_last = randint(5, 10)  # 5s ~ 10s
        print(f'{datetime.now()} - Waiting to next json after {wait_time_last} seconds ...')
        await asyncio.sleep(wait_time_last)

# **執行 `main()`**，確保程式運行在 **異步模式**
asyncio.run(main())
