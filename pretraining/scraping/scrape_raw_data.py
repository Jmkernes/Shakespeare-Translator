import time
import requests
from bs4 import BeautifulSoup
import tensorflow as tf


### Raw data preprocessing functions
def unicode_to_ascii_quotes(text):
    text = tf.strings.regex_replace(text, b'\xe2\x80\x99', b"'")
    text = tf.strings.regex_replace(text, b'\xe2\x80\x98', b"'")
    text = tf.strings.regex_replace(text, b'\xe2\x80\x9d', b"\"")
    text = tf.strings.regex_replace(text, b'\xe2\x80\x9c', b"\"")
    text = tf.strings.regex_replace(text, b'\xe2\x80\x93', b", ")
    text = tf.strings.regex_replace(text, b'\xe2\x80\x94', b", ")
    return text

def remove_whitespace(text):
    text = tf.strings.regex_replace(text, b'\s+', b' ')
    text = tf.strings.strip(text)
    return text

def insert_line_breaks(text):
    text = tf.strings.regex_replace(text, '\Q.\E', '.<LINE>')
    text = tf.strings.regex_replace(text, '\Q?\E', '?<LINE>')
    text = tf.strings.regex_replace(text, '\Q!\E', '!<LINE>')
    text = tf.strings.regex_replace(text, '\Q;\E', ';<LINE>')
    return text

def flatten_dataset(ds, marker='<LINE>'):
    all_data = []
    for line in ds:
        if tf.strings.length(line):
            all_data.append(line)
    all_data = tf.strings.split(all_data, sep=marker).flat_values
    return tf.data.Dataset.from_tensor_slices(all_data)

def combine_lines(ds, n):
    ds = ds.window(n, n, 1).flat_map(lambda w: w.batch(n))
    ds = ds.map(lambda x: tf.strings.reduce_join(x, separator=b' '))
    ds = ds.map(remove_whitespace)
    ds = ds.map(tf.strings.strip)
    return ds

def preprocess(ds, n):
    ds = ds.map(unicode_to_ascii_quotes)
    ds = ds.map(remove_whitespace)
    ds = ds.map(insert_line_breaks)
    ds = flatten_dataset(ds)
    ds = combine_lines(ds, n)
    return ds

def main():

    ### Download the 2020 best reads
    URL = 'https://longreads.com/topics/'
    page = requests.get(URL)

    soup = BeautifulSoup(page.content, 'html.parser')
    num = 1

    start = time.time()
    print("Creating longreads2020 text file.")
    with open('longreads2020.txt', 'w') as file:
        for link in soup.find_all(class_='grid-title')[1:]:
            link = link.find('a')['href']
            print(f"Loading article {num}: {link}")
            middle_page = requests.get(link)
            middle_soup = BeautifulSoup(middle_page.content, 'html.parser')
            real_link = middle_soup.find_all(class_='button-red')
            real_link = real_link[-1]

            if real_link.get_text() in ['Read the story','Read the essay','Read the interview']:
                real_link = real_link['href']
                print(f"\tOpening real page: {real_link}")
                real_page = requests.get(real_link)
                real_soup = BeautifulSoup(real_page.content, 'html.parser')
                text = real_soup.find_all('p')
            else:
                print("Could not find a valid out link. Printing current contents.")
                text = middle_soup.find_all('p')

            print(f"\tWriting {len(text)} lines to file...")
            if len(text) < 5:
                print("Fewer than 5 lines available. Skipping.")
                continue
            for line in text:
                file.write(line.get_text())
            num += 1
            curr_time = time.time()-start
            print(f"\tCurrent time: {curr_time:.2f}. {curr_time/num:.2f}pg/s")
            print('-'*50)


    ### Download the 2019 best reads
    URL = ('https://longreads.com/2019/12/09/'
        'longreads-best-of-2019-all-of-our-no-1-story-picks/')

    page = requests.get(URL)

    soup = BeautifulSoup(page.content, 'html.parser')
    num = 1

    print("Creating longreads2019 text file.")
    start = time.time()
    with open('longreads2019.txt', 'w') as file:
        for link in soup.find_all(class_='top5-title'):
            link = link.find('a')['href']
            print(f"Loading article {num}: {link}")
            real_page = requests.get(link)
            real_soup = BeautifulSoup(real_page.content, 'html.parser')
            text = real_soup.find_all('p')

            print(f"\tWriting {len(text)} lines to file...")
            if len(text) < 5:
                print("Fewer than 5 lines available. Skipping.")
                continue
            for line in text:
                file.write(line.get_text())
            num += 1
            curr_time = time.time()-start
            print(f"\tCurrent time: {curr_time:.2f}. {curr_time/num:.2f}pg/s")
            print('-'*50)

    ### 2018
    URL = 'https://longreads.com/2018/'
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')
    num = 1
    start = time.time()
    print("Creating longreads2018 text file.")
    with open('longreads2018.txt', 'w') as file:
        for link in soup.find_all(class_='entry-title')[1:]:
            link = link.find('a')['href']
            print(f"Loading article {num}: {link}")
            middle_page = requests.get(link)
            middle_soup = BeautifulSoup(middle_page.content, 'html.parser')
            real_link = middle_soup.find_all(class_='button-red')
            real_link = real_link[-1]

            if real_link.get_text() in ['Read the story','Read the essay','Read the interview']:
                real_link = real_link['href']
                print(f"\tOpening real page: {real_link}")
                real_page = requests.get(real_link)
                real_soup = BeautifulSoup(real_page.content, 'html.parser')
                text = real_soup.find_all('p')
            else:
                print("Could not find a valid out link. Printing current contents.")
                text = middle_soup.find_all('p')

            print(f"\tWriting {len(text)} lines to file...")
            if len(text) < 5:
                print("Fewer than 5 lines available. Skipping.")
                continue
            for line in text:
                file.write(line.get_text())
            num += 1
            curr_time = time.time()-start
            print(f"\tCurrent time: {curr_time:.2f}. {curr_time/num:.2f}pg/s")
            print('-'*50)


    ### 2017
    URL = 'https://longreads.com/2017/12/29/the-25-most-popular-longreads-exclusives-of-2017/'
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')
    num = 1
    start = time.time()
    print("Creating longreads2017 text file.")
    with open('longreads2017.txt', 'w') as file:
        for link in soup.find_all(class_='top5-title')[1:]:
            link = link.find('a')['href']
            print(f"Loading article {num}: {link}")
            middle_page = requests.get(link)
            middle_soup = BeautifulSoup(middle_page.content, 'html.parser')
            real_link = middle_soup.find_all(class_='button-red')
            real_link = real_link[-1]

            if real_link.get_text() in ['Read the story','Read the essay','Read the interview']:
                real_link = real_link['href']
                print(f"\tOpening real page: {real_link}")
                real_page = requests.get(real_link)
                real_soup = BeautifulSoup(real_page.content, 'html.parser')
                text = real_soup.find_all('p')
            else:
                print("Could not find a valid out link. Printing current contents.")
                text = middle_soup.find_all('p')

            print(f"\tWriting {len(text)} lines to file...")
            if len(text) < 5:
                print("Fewer than 5 lines available. Skipping.")
                continue
            for line in text:
                file.write(line.get_text())
            num += 1
            curr_time = time.time()-start
            print(f"\tCurrent time: {curr_time:.2f}. {curr_time/num:.2f}pg/s")
            print('-'*50)

    ### Process data and write to file
    for x in ['longreads2020.txt', 'longreads2019.txt', 'longreads2018.txt',
    'longreads2017.txt', 'modern.txt']:
        ds = tf.data.TextLineDataset(x)
        ds = preprocess(ds, 2)
        with open('modern_corpus.txt', 'a+') as file:
            for line in ds:
                line = line.numpy().decode()
                file.write(line)
                file.write('\n')

if __name__=="__main__":
    main()
