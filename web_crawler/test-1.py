"""
暂时废弃爬虫的尝试 2025-10-02
"""










# import asyncio
# from crawl4ai import AsyncWebCrawler

# async def main():
#     async with AsyncWebCrawler() as crawler:#在异步上下文管理器（async context manager）中创建并使用一个 AsyncWebCrawler 实例，把进入上下文时返回的对象绑定到变量 crawler。
#         result = await crawler.arun("https://example.com")
#         print(result.markdown[:300])  # Print first 300 chars

# if __name__ == "__main__":
#     asyncio.run(main())


# 官方文档地址：https://docs.crawl4ai.com/core/quickstart/
# 命令行使用：
# # Basic crawl with markdown output
# crwl https://www.nbcnews.com/business -o markdown

# # Deep crawl with BFS strategy, max 10 pages
# crwl https://docs.crawl4ai.com --deep-crawl bfs --max-pages 10

# # Use LLM extraction with a specific question
# crwl https://www.example.com/products -q "Extract all product prices"


# import asyncio
# from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

# async def main():
#     browser_conf = BrowserConfig(headless=False)  # or False to see the browser
#     run_conf = CrawlerRunConfig(
#         cache_mode=CacheMode.BYPASS
#     )

#     async with AsyncWebCrawler(config=browser_conf) as crawler:
#         result = await crawler.arun(
#             url="https://example.com",
#             config=run_conf
#         )
#         print(result.markdown)

# if __name__ == "__main__":
#     asyncio.run(main())

from crawl4ai import AsyncWebCrawler, AdaptiveCrawler
import asyncio


async def main():
    async with AsyncWebCrawler() as crawler:
        # Create an adaptive crawler (config is optional)
        adaptive = AdaptiveCrawler(crawler)

        # Start crawling with a query
        result = await adaptive.digest(
            start_url="https://www.google.com/search?sca_esv=9379877597def705&sxsrf=AE3TifPgtXm0NW_x1O2xzNgcGsMdfS52kA:1759068864643&udm=2&fbs=AIIjpHybaGNnaZw_4TckIDK59Rtx7EmYoHRazOl26McMSIhENyiO40OXF-2AmuvvRc2crJX2_4Hltwod39Ayr9WMohoOob8oumIXOAxq_A-qcl2uIrzn4b1RMnikPVjWrn5UusY2uCBn15YIkGKRel1IBJ_BZ0bqIbYX4VfJZIID_R-cyafnuW2-ZMGHrB98qLnYw10cQx_0WzaoWh9Ltuf3nAy0g86QwQ&q=%E7%8B%97&sa=X&ved=2ahUKEwj22L-W0vuPAxXJevUHHYznL_UQtKgLegQIHhAB&biw=960&bih=1538&dpr=1.5",
            query="images of dogs"
        )

        # View statistics
        adaptive.print_stats()

        # Get the most relevant content
        relevant_pages = adaptive.get_relevant_content(top_k=20)
        for page in relevant_pages:
            print(f"- {page['url']} (score: {page['score']:.2f})")

if __name__ == "__main__":
    asyncio.run(main())
