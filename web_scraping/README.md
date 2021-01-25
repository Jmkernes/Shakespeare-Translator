# Web scraping

This directory is not necessary for running and interacting with the project. It exists to show how the dataset was extracted, and to allow you to scrape it from the web if you would like to try it from scratch!

The main file is web_scraper.py. It takes 3 flags:

1) output directory to write translations to

2) the name of a log file to output scraping status to

3) skip: how many plays to skip reading (will explain in a second)

The program will scrape shakespeare text and its modern translation and write it to a series of text files.

## Data structure and source

The data comes from sparknotes' "no fear shakespeare". For every play, they offer side by side comparisons of original shakespeare text plus a modern translation. These are grouped via paired HTML elements, making scraping possible.

Each shakespeare work is broken down first by the play (obviously. Plays like romeo and juliet, julius caesar, etc.), then by Act number, then by Scene number. Each individual webpage will consist of one scene with a hand full of translations.

The original text contains line numbers [35] etc. that need to be removed. There are also a number of names indicating which character is speaking (e.g. CLEOPATRA:) that need to be dealt with. Finally, there are stage direction lines like "exeunt" or "characters leave stage" that should be skipped.

## the log file
Since we are scraping, we run the risk of getting blocked. This will abruptly kill the scraping process. to save progress, we write one play file at a time. Furthermore, we simultaneously write to a log file log.txt logging anytime a successful scrape was made to avoid re-doing a scrape. The "skip" flag, allows us to skip over a certain number of plays, so that we don't re-scrape when the code gets stalled

## Possible issues
There are a few minor issues that arise when scraping. First, we scrape by incremementing the page number part of the html .../page1 .../page2 etc. Some plays follow strange page schemes (henryvpt2 is one, as it is a sequel it does not begin at page 1). These issues are mentioned in the bottom of the log file.

## Happy scraping!
