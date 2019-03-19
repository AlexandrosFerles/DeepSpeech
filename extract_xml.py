import xml.etree.ElementTree as etree
import string

def extract_xml(name):

    coutner_sentences = 0
    sentence = ''

    f = open('/home/guest/Desktop/corpus_.txt', 'w')

    # test = etree.iterparse(name, events=('start', 'end'))

    for event, elem in etree.iterparse(name, events=('start', 'end')):

        if event == 'start' and elem.tag == 'w':
            if elem.text == None:
                continue

            sentence += elem.text + ' '
            elem.clear()

        if event == 'end' and elem.tag == 'sentence':

            if coutner_sentences % 1000 == 0:
                print(coutner_sentences)
            coutner_sentences += 1

            sentence = sentence.lower()
            sentence = sentence.replace(',', ' ').replace('è', "e").replace('é', "e").replace("ÿ", "y").replace("ü", "u")

            for c in string.punctuation:
                sentence = sentence.replace(c, " ")

            f.write(sentence + '\n')

            sentence = ''
            elem.clear()

    f.close()

if __name__=='__main__':

    extract_xml(name='/home/guest/Downloads/familjeliv-allmanna-familjeliv.xml')
