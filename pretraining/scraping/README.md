# Web scraping

The original english translatio of the shakespeare corpus is around 30k lines. In order to improve results, we implement a masked language model (LM) pretraining routine to learn a good set of token embeddings. To boost the efficacy of this semi-supervised learning task, we supplement the original data with additional english language text, around an additional 30k lines.

We acquire more data by web scraping for long english-language articles. To do this, we scrape from the website longreads.com. We chose this for a few reasons.

1) Longer articles ==> more easily accessible data with higher likelihood for consecutive sentences to be semantically linked

2) Longreads doesn't have original content, they just compile links to many external sites. Scraping is much easier because we are requesting a wide variety of sites, preventing any request blocking from overloading one particular address.

3) The articles cover a variety of topics written by different authors, so hopefully they give the necessary depth.

The actual script to scrape is given in web_scraper.py. The results of that script are in the text files longreads\*, where the number indicates they were pulled from a "best of the year \_\_\_" list.

## Data cleaning.

In order to get an easily digestible dataset, we want to split the text into piecs roughly the size of our sequence length. To do this, we flatten out all of the data, then insert linebreaks following periods, question marks, exclamation points and semi-colons. We then window the data into pairs of sentences, and write each pair to a single line in the final output file (called modern\_corpus.txt). The same thing is applied to the original shakespeare dataset as well

This procedure gets us sequences that are mostly around 100 tokens, with not too many much shorter or much longer.
