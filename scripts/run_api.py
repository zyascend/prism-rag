#!/usr/bin/env python
"""启动 API 服务"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("src.api.routes:app", host="0.0.0.0", port=8000, reload=False)