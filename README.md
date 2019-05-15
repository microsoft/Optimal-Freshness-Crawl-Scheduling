# Introduction
This archive contains the dataset used in the empirical evaluation of the following paper:

A. Kolobov, E. Lubetzky, Y. Peres, E. Horvitz, "Optimal Freshness Crawl Under Politeness Constraints", SIGIR-2019.

When using the dataset in other works, please cite the article above.

We thank Microsoft Bing for help in collecting this data. For any questions regarding the dataset, please contact Andrey Kolobov (akolobov@microsoft.com, https://www.microsoft.com/en-us/research/people/akolobov/).

# Data Collection Details
The dataset was gathered by crawling a large collection of URLs for approximately 14 weeks in 2017 using Microsoft Bing's production web crawler, and upon every crawl recording whether the corresponding web page has changed since its previous crawl. These URLs were used as sources of structured information, e.g., event times, for Microsoft's Satori knowledge base. For this purpose, information of interest was extracted from page content using templates. Accordingly, we considered a URL as changed across two crawls if and only if:

- Both of the crawls succeeded (HTTP status code < 400) and the information extracted from the two page versions using the templates was different, or

- One of the crawls succeeded but the other one yielded a 4xx HTTP status code.

The crawler was instructed to visit each URL from this collection approximately once a day. However, factors ranging from spikes in the crawler's production workload to temporary host unavailability caused some URL crawl requests on some days to be dropped or otherwise fail. We didn't record these crawl requests in the dataset, so for a given URL the number of recorded crawls can differ from 98, the expected number for 14 weeks' worth of daily crawls.

At the same time, a small number of URLs were crawled far more frequently than once a day due to production crawl requests.

After completing the 14-week crawl, we dropped the URLs that:

- Weren't crawled successfully a single time over 14 weeks, either due to their crawls failing due to crawler's internal reasons or due to 4xx status codes.
 
- Didn't change (according to the above definition of change) a single time over 14 weeks. We assumed that these URLs never change, and therefore the algorithm from the aforementioned SIGIR-2019 paper would ignore them for the purposes of crawl scheduling.

- Are located on hosts for which Bing's crawler is uncertain about politeness constraints. In contrast, for the host of each URL that appears in the dataset, Bing's crawler either has a high-confidence estimate of its politeness constraint for Bing or is highly confident that this host isn't imposing such a constraint.

Bing assigns "importance scores" to the URLs it crawls. They are a combination of Bing's PageRank-like measure and a click-based URL popularity value. For each URL in the dataset, we recorded the importance score Bing assigned to it as of the start of the 14-week crawl. The higher the score, the more important the Bing considers the URL.

# Dataset Format
The resulting dataset contains importance scores, host information and 14-week crawl history for ~18M URLs, broken down across several TSV files:

-- url_urlid_hostid.txt. Its columns are 

* URL
* URL_ID -- contains a unique ID for each crawled URL in the dataset, used as key in other dataset files
* Host_ID -- the ID corresponding to this URL's host


-- host_hostid_isconstrained.txt. Its columns are 

* Host_URL -- contains URLs of each host on which at least one crawled URL from our dataset is located
* Host_ID -- contains a unique ID for each host in the dataset, used as key in other dataset files
* Is_constrained? -- contains a 1 for each host that imposes a politeness constraint on Bing's crawler, 0 otherwise


-- urlid_imp.txt. Its columns are

* URL_ID -- see url_urlid_hostid.txt
* Importance_score -- the URL's Bing-assigned importance score


-- urlid_offset_history.txt. Its columns are

* URL_ID -- see url_urlid_hostid.txt
* First_crawl_time_offset -- length of the time interval, in days, between the moment (common for all URLs in the dataset) when the 14-week data collection started and the first time this URL was actually crawled.
* Crawl_and_change_history -- an ordered list of [time_offset, change_indicator] pairs, where time_offset is the length of the time interval, in days, since this URL's previous crawl, and change_indicator = 1 if the URL has change since the previous crawl, 0 otherwise. For example, row

5	5.5143055555555556	[[1.10396990740741, 0], [1.47311342592593, 1], ...

indicates that URL with ID=5 in url_urlid_hostid.txt was first crawled ~5.5 days since data collection started. ~1.1 days later it was crawled again but didn't change compared to the first crawl. ~1.5 days after the second crawl it was crawled yet again, and was discovered to have changed compared to the second crawl, etc.

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
