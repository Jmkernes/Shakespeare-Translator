import os
import re
import csv
import tqdm
import time
import requests
from bs4 import BeautifulSoup
# from pdb import set_trace as bp

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('data_directory', 'raw_data', 'directory to write data to')
flags.DEFINE_string('log_file', 'log.txt', 'name of log file to write to')
flags.DEFINE_integer('play_number', 0, 'The play number to skip to.',
                    lower_bound=0, upper_bound=24)

def standardize(s):
    """Remove numbers, newlines, and trailing/leading whitespaces
    Parameters:
        s (str): a python string.
    Outputs:
        (str) the reformatted string.
    """
    s = re.sub('\d+', ' ', s)
    s = re.sub('\n', ' ', s)
    s = re.sub(r'  +', ' ', s)
    s = s.strip()
    return s

def extractPage(page):
    """Extract original text and its associated modern translation
    Parameters:
        page (requests object): a requests.get(URL) object.
    Outputs:
        None (NoneType): returns None if parsing the page encountered an error.
        Possible errors include original-modern shape mismatch, empty
        strings, or a failure to find the page elements.

        data (List(Tuple(str,str))): returns a list of
        tuples (x_1, y_1), (x_2, y_2),... Where x is the original text
        string, and y is the translation string.
    """
    soup = BeautifulSoup(page.content, 'html.parser')

    # group by cell to ensure equal parts original and modern text
    original_text, modern_text = [], []
    original_cells = soup.find_all(
        'td', class_='noFear__cell noFear__cell--original')
    modern_cells = soup.find_all(
        'td', class_='noFear__cell noFear__cell--modern')

    if not original_cells or not modern_cells:
        print("No data on this page. Skipping...")
        return None

    for cell in original_cells:
        curr = []
        lines = cell.find_all(
            'div', class_='noFear__line noFear__line--original')
        # sometimes cells with page numbers get a different tag.
        if not lines:
            lines = cell.find_all(
                'div', class_=("noFear__line noFear__line"
                               "--original noFear__line--hasLineNumber"))
        if not lines:
            continue
        for line in lines:
            curr.append(standardize(line.get_text()))
        original_text.append(' '.join(curr))

    for cell in modern_cells:
        curr = []
        lines = cell.find_all('div',
                              class_='noFear__line noFear__line--modern')
        if not lines:
            continue
        for line in lines:
            curr.append(standardize(line.get_text()))
        modern_text.append(' '.join(curr))

    if len(original_text)!=len(modern_text):
        print("Error, length mismatch. Skipping page...")
        return None
    return list(zip(original_text, modern_text))

def printBar(filename, page, max_page_num, diff):
    """ Prints the current extraction status, pages scraped,
    and the time per page extraction.
    Parameters:
        filename (str): a string giving the name of the current Shakespearian
        play being extracted.

        page (int): an integer giving the current page number being extracted.

        max_page_num (int): total number of pages to scrape. Defaults to MAX_PAGE_NUMBER.

        diff (float): a float giving the average extraction time per page so far.
    Outputs:
        (void) prints to standard output stream.
    """
    num_eq = int(20.0*page/max_page_num)
    num_dot = 20-num_eq-1
    middle = '['+'='*num_eq+'>'+'.'*num_dot+']'
    end = '\n' if page==max_page_num-1 else '\r'
    print(f"Writing file {filename}. ",
          f"Page: {page+1}/{max_page_num}.",
          middle,
          f"   {diff:.3f}: sec./page",
          end=end)

def scrape(data_directory, log_file, PLAY_URLS, skip=0):
    """ Scrapes translations from no fear shakespeare website. Writes the data
    to a specified directory and outputs a log file detailing successful extractions
    Parameters:
        data_directory (str): the specified directory to write data to. Writes
        in CSV format.

        log_file (str): file path for the log.txt file.

        skip (int): an integer 0<skip<25 indicating which play number to begin with.
        This is used if the scraping was halted for some reason. Look at the log_file
        And choose the play number of the last unsuccessful extraction
    Outputs:
        (void): directly writes CSV files to the data_directory
    """
    MAX_PAGE_NUMBER = 700
    start = time.time()
    skip = 0 # a parameter to begin scraping later in case of IP blocking
    for play_num, play_url in enumerate(PLAY_URLS[skip:]):
        play_num += skip
        name = play_url[46:-1]
        max_page_num=MAX_PAGE_NUMBER
        MIN_PAGE_NUMBER=254 if name=='henry4pt2' else 0
        print('-'*10,f" Writing data from play number {play_num}: {name} ",'-'*10)
        start_play = time.time()
        with open(data_directory+name, 'w') as csvfile:
            writer = csv.writer(csvfile)
            num_fails = 0   # Count number of consecutive 404 errors
            for page_num in range(MIN_PAGE_NUMBER, MAX_PAGE_NUMBER):

                if num_fails > 4: # Stop searching after 5 fails.
                    num_fails=0
                    break

                if name=='sonnets':
                    page_url = play_url+'sonnet_'+str(page_num)+'/'
                else:
                    page_url = play_url+'page_'+str(page_num)+'/'
                page = requests.get(page_url)

                if not page.ok:
                    max_page_num = page_num-num_fails-1
                    num_fails += 1
                    continue

                num_fails = 0
                data = extractPage(page)
                if data is None:
                    continue
                for x, y in data:
                    if x and y:
                        writer.writerow([x,y])

                diff = (time.time()-start_play)/(page_num+1)
                printBar(name, page_num, MAX_PAGE_NUMBER, diff)

        diff = time.time()-start_play
        filesize = os.path.getsize(os.path.join(os.getcwd(), name))
        filesize /= 1e6
        print(f"Finished writing... Time: {diff:.2f}s. File size: {filesize:.2f}MB\n")

        # Since the play was successfully extracted, write this to a log file
        with open(log_file, 'a') as file:
            file.write('-'*10+" Writing data from play number {}: {} ".format(play_num, name)+'-'*10+'\n')
            file.write((f"Writing file {name}. Page: {max_page_num}/{max_page_num}."
                       f"[============>]   {diff/(max_page_num):.2f}: sec./page\n"))
            file.write(f"Finished writing... Time: {diff:.2f}s. File size: {filesize:.2f}MB\n\n")

    print("~"*10, " Finished writing all files ", "~"*10)
    print(f"Time: {time.time()-start}")

def getPlayURLS():
    ROOT = 'https://www.sparknotes.com'
    URL = 'https://www.sparknotes.com/shakespeare'
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')
    plays = soup.find_all('a', class_="no-fear-thumbs__thumb__link")
    # HOME = 'https://www.sparknotes.com/nofear/shakespeare/'
    PLAY_URLS = [ROOT+link.get('href') for link in plays]
    return PLAY_URLS

def createLogFile(filename):
    with open(filename, 'w') as file:
        s = '~'*15+" Log file "+'~'*15
        file.write("*"*len(s)+'\n')
        file.write(s+'\n')
        file.write("*"*len(s)+'\n'+'\n')

def main(argv):
    skip = FLAGS.play_number
    data_directory = FLAGS.data_directory
    log_file = FLAGS.log_file
    if skip==0:
        print(f"Creating log file at {log_file}...")
        createLogFile(log_file)
    else:
        print(f"Log file already exists...")
    try:
        os.mkdir(data_directory)
        print(f"Creating directory {data_directory}...")
    except:
        print(f"Directory \"{data_directory}\" already exists...")
    if data_directory[-1]!='/':
        data_directory += '/'
    PLAY_URLS = getPlayURLS()
    scrape(data_directory, log_file, PLAY_URLS, skip)

if __name__=="__main__":
    app.run(main)
