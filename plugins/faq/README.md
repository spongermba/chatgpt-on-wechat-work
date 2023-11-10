    uvicorn main:app --host '0.0.0.0' --port 8088 --reload

    ```
    以守护进程的方式一直运行
    nohup uvicorn main:app --host '0.0.0.0' --port 8088 &