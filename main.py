import os
import pickle
import argparse
import numpy as np
from weighting_schemes import checkDictionary, sentence_length_weight
from utils_new import cosine_metrix, euclidean_metrix, metriclearning_metrix, combined_metrix, get_MBS_metrix

wordDictionary = {}

def main():

    parser = argparse.ArgumentParser(
        'Align sentences',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-l', '--language_pair', type=str, required=True,
                        choices = ['si-en', 'ta-en', 'si-ta'],
                        help='source and target languages separated by a hyphen')

    parser.add_argument('-w', '--website', type=str, required=True,
                        choices = ['armynews' ,'hiru', 'itn', 'newsfirst'],
                        help='name of the website')

    parser.add_argument('-s', '--similarity_measure', type=str, required=True,
                        choices = ['cosine' ,'euclidean', 'sdml', 'itml', 'combined_sdml', 'combined_itml'],
                        help='Sentence similarity measurement.')

    parser.add_argument('-r', '--ratio', type=bool, default=False,
                        help='Use ratio score or not.')

    parser.add_argument('-d', '--dictionary', type=bool, default=False,
                        help='Use dictionary weighting or not.')

    args = parser.parse_args()
    language_pair = args.language_pair
    website = args.website
    similarity_measure = args.similarity_measure
    ratio = args.ratio
    dictionary = args.dictionary

    get_alignment(language_pair, website, similarity_measure, ratio, dictionary)
    get_intersection(language_pair, website, similarity_measure)
    recall(language_pair, website, similarity_measure)

def get_alignment(language_pair, website, similarity_measure, ratio, dictionary):

    wordDictionary = loadDictionary(language_pair)
    sdml = load_sdml_model(language_pair)
    itml = load_itml_model(language_pair)

    # path_forward_alignment = "/media/laka/Lakmali/FYP/dilan_paper/"+ language_pair + "/" + website + "/" + similarity_measure + ".forward"
    # path_backward_alignment = "/media/laka/Lakmali/FYP/dilan_paper/"+ language_pair + "/" + website + "/" + similarity_measure + ".backward"
    path_forward_alignment = "/media/laka/Lakmali/FYP/dilan_paper/" + similarity_measure + ".forward"
    path_backward_alignment = "/media/laka/Lakmali/FYP/dilan_paper/" + similarity_measure + ".backward"
    forward_alignment = open(path_forward_alignment, "a")
    backward_alignment = open(path_backward_alignment, "a")

    root_dir = "/media/laka/Lakmali/FYP/Data_sets/dilan_ppr/" + language_pair + "/" + website
    if language_pair == 'si-en':
        embeddings_A = root_dir + "/embedding/Sinhala/"
        embeddings_B = root_dir + "/embedding/English/"
        docs_A = root_dir + "/sentences/Sinhala/"
        docs_B = root_dir + "/sentences/English/"

    if language_pair == 'ta-en':
        embeddings_A = root_dir + "/embedding/Tamil/"
        embeddings_B = root_dir + "/embedding/English/"
        docs_A = root_dir + "/sentences/Tamil/"
        docs_B = root_dir + "/sentences/English/"

    if language_pair == 'si-ta':
        embeddings_A = root_dir + "/embedding/Sinhala/"
        embeddings_B = root_dir + "/embedding/Tamil/"
        docs_A = root_dir + "/sentences/Sinhala/"
        docs_B = root_dir + "/sentences/Tamil/"

    for file in os.listdir(embeddings_A):
        path_embA = embeddings_A + file
        path_embB = embeddings_B + file
        path_docA = docs_A + file
        path_docB = docs_B + file
        embA = read_emb_file(path_embA)
        embB = read_emb_file(path_embB)
        sentences_A = read_sentence_file(path_docA)
        sentences_B = read_sentence_file(path_docB)

        if similarity_measure == 'cosine':
            metrix_AB = cosine_metrix(embA, embB)
        elif similarity_measure == 'euclidean':
            metrix_AB = euclidean_metrix(embA, embB)
        elif similarity_measure == 'sdml':
            metrix_AB = np.array(metriclearning_metrix(embA, embB, sdml))
        elif similarity_measure == 'itml':
            metrix_AB = np.array(metriclearning_metrix(embA, embB, itml))
        elif similarity_measure == 'combined_sdml':
            cosine = cosine_metrix(embA, embB)
            euclidean = euclidean_metrix(embA, embB)
            metriclearning = metriclearning_metrix(embA, embB, sdml)
            metrix_AB = combined_metrix(cosine, euclidean, metriclearning)
        elif similarity_measure == 'combined_itml':
            cosine = cosine_metrix(embA, embB)
            euclidean = euclidean_metrix(embA, embB)
            metriclearning = metriclearning_metrix(embA, embB, itml)
            metrix_AB = combined_metrix(cosine, euclidean, metriclearning)

        if dictionary:
            if ratio:
                metrix_A = []
                for index_a in range(len(metrix_AB)):
                    weighted_similarities = []
                    for index_b in range (len(metrix_AB.T)):
                        siLine = sentences_A[index_a][1]
                        enLine = sentences_B[index_b][1]
                        overlap_count = checkDictionary(siLine, enLine, wordDictionary)[1]
                        weight_dict = len(siLine.split()) / (len(siLine.split()) + 1 - overlap_count)
                        weighted_similarities.append(metrix_AB[index_a, index_b] * weight_dict)
                    metrix_A.append(weighted_similarities)
                metrix_A = np.array(metrix_A)

                metrix_B = []
                for index_b in range(len(metrix_AB.T)):
                    weighted_similarities = []
                    for index_a in range(len(metrix_AB)):
                        siLine = sentences_A[index_a][1]
                        enLine = sentences_B[index_b][1]
                        overlap_count = checkDictionary(siLine, enLine, wordDictionary)[0]
                        try:
                            weight_dict = len(enLine.split()) / (len(enLine.split()) + 1 - overlap_count)
                        except:
                            weight_dict = len(enLine.split())
                        weighted_similarities.append(metrix_AB[index_a, index_b] * weight_dict)
                    metrix_B.append(weighted_similarities)
                metrix_B = np.array(metrix_B)

                metrix_AB = np.add(metrix_A, metrix_B.T)/2
                metrix_AB = get_MBS_metrix(metrix_AB, embA, embB)

                for i in range(len(metrix_AB)):
                    index_a = i
                    index_b = np.argmax(metrix_AB[i])
                    forward_alignment.write(sentences_A[index_a][0] + "\t" + sentences_B[index_b][0] + "\n")
                for i in range(len(metrix_AB.T)):
                    index_b = i
                    index_a = np.argmax(metrix_AB.T[i])
                    backward_alignment.write(sentences_A[index_a][0] + "\t" + sentences_B[index_b][0] + "\n")

            else:
                for index_a in range(len(embA)):
                    similarities = metrix_AB[index_a]
                    max_indexes = np.argsort(similarities)[-4:]
                    weighted_similarities = []
                    for i in max_indexes:
                        siLine = sentences_A[index_a][1]
                        enLine = sentences_B[i][1]
                        overlap_count = checkDictionary(siLine, enLine, wordDictionary)[1]
                        weight_dict = len(siLine.split()) / (len(siLine.split()) + 1 - overlap_count)
                        weighted_similarities.append(similarities[i] * weight_dict)
                    max_score = np.amax(weighted_similarities)
                    index_max = np.argmax(weighted_similarities)
                    index_b = max_indexes[index_max]
                    forward_alignment.write(sentences_A[index_a][0] + "\t" + sentences_B[index_b][0] + "\n")

                for index_b in range(len(embB)):
                    similarities = metrix_AB.T[index_b]
                    max_indexes = np.argsort(similarities)[-4:]
                    weighted_similarities = []
                    for i in max_indexes:
                        siLine = sentences_A[i][1]
                        enLine = sentences_B[index_b][1]
                        overlap_count = checkDictionary(siLine, enLine, wordDictionary)[0]
                        try:
                            weight_dict = len(enLine.split()) / (len(enLine.split()) + 1 - overlap_count)
                        except:
                            weight_dict = len(enLine.split())
                        weighted_similarities.append(similarities[i] * weight_dict)
                    max_score = np.amax(weighted_similarities)
                    index_max = np.argmax(weighted_similarities)
                    index_a = max_indexes[index_max]
                    backward_alignment.write(sentences_A[index_a][0] + "\t" + sentences_B[index_b][0] + "\n")

        else:
            if ratio:
                metrix_AB = get_MBS_metrix(metrix_AB, embA, embB)

            for i in range (len(metrix_AB)):
                index_a = i
                index_b = np.argmax(metrix_AB[i])
                forward_alignment.write(sentences_A[index_a][0] + "\t" + sentences_B[index_b][0] + "\n")
            for i in range (len(metrix_AB.T)):
                index_b = i
                index_a = np.argmax(metrix_AB.T[i])
                backward_alignment.write(sentences_A[index_a][0] + "\t" + sentences_B[index_b][0] + "\n")

    forward_alignment.close()
    backward_alignment.close()


def read_sentence_file(path):
    sentences = open(path, "r").read()
    sentences = sentences.split("\n")
    sentences = sentences[:-1]
    sentences = [i.split("\t") for i in sentences]
    return sentences


def read_file(path):
    sentences = open(path, "r").read()
    sentences = sentences.split("\n")
    sentences = sentences[:-1]
    return sentences


def read_emb_file(path):
    dim = 1024
    X = np.fromfile(path, dtype=np.float32, count=-1)
    X.resize(X.shape[0] // dim, dim)
    return X


def get_intersection(language_pair, website, similarity_measure):
    # path_forward = "/media/laka/Lakmali/FYP/dilan_paper/"+ language_pair + "/" + website + "/" + similarity_measure + ".forward"
    # path_backward = "/media/laka/Lakmali/FYP/dilan_paper/"+ language_pair + "/" + website + "/" + similarity_measure + ".backward"
    # path_intersection = "/media/laka/Lakmali/FYP/dilan_paper/"+ language_pair + "/" + website + "/" + similarity_measure + ".intersection"
    path_forward = "/media/laka/Lakmali/FYP/dilan_paper/" + similarity_measure + ".forward"
    path_backward = "/media/laka/Lakmali/FYP/dilan_paper/" + similarity_measure + ".backward"
    path_intersection = "/media/laka/Lakmali/FYP/dilan_paper/" + similarity_measure + ".intersection"
    forward = read_file(path_forward)
    backward = read_file(path_backward)
    intersection = open(path_intersection, "a")
    for i in range(len(forward)):
        if forward[i] in backward:
            intersection.write(forward[i] + "\n")
    intersection.close()


def recall(language_pair, website, similarity_measure):
    # path_forward = "/media/laka/Lakmali/FYP/dilan_paper/"+ language_pair + "/" + website + "/" + similarity_measure + ".forward"
    # path_backward = "/media/laka/Lakmali/FYP/dilan_paper/"+ language_pair + "/" + website + "/" + similarity_measure + ".backward"
    # path_intersection = "/media/laka/Lakmali/FYP/dilan_paper/"+ language_pair + "/" + website + "/" + similarity_measure + ".intersection"
    path_forward = "/media/laka/Lakmali/FYP/dilan_paper/" + similarity_measure + ".forward"
    path_backward = "/media/laka/Lakmali/FYP/dilan_paper/" + similarity_measure + ".backward"
    path_intersection = "/media/laka/Lakmali/FYP/dilan_paper/" + similarity_measure + ".intersection"
    path_golden = "/media/laka/Lakmali/FYP/Data_sets/dilan_ppr/"+ language_pair + "/" + website + "/" + website + "." + language_pair
    forward = read_file(path_forward)
    backward = read_file(path_backward)
    intersection = read_file(path_intersection)
    golden = read_file(path_golden)
    print ("---------------Forward---------------------")
    cal_recall(golden, forward)
    print ("---------------Backward--------------------")
    cal_recall(golden, backward)
    print ("---------------Intersection----------------")
    cal_recall(golden, intersection)


def cal_recall(golden, alignment):
    match_count = 0
    for i in golden:
        if i in alignment:
            match_count += 1

    recall = (match_count / len(golden)) * 100
    print ("Recall :", recall)
    print ("Matches :", match_count)


def loadDictionary(language_pair):

    root_dir = "/media/laka/Lakmali/FYP/dilan_paper/scripts/Dictionaries/" + language_pair

    if language_pair == 'si-en':
        dictionaryA = root_dir + "/combineddictionary.en"
        dictionaryB = root_dir + "/combineddictionary.si"
        # dictionaryA = root_dir + "/existingdictionary.en"
        # dictionaryB = root_dir + "/existingdictionary.si"
        personNamesA = root_dir + "/person-names.en"
        personNamesB = root_dir + "/person-names.si"
        designationsA =  root_dir + "/designations.en"
        designationsB = root_dir + "/designations.si"

    if language_pair == 'ta-en':
        dictionaryA = root_dir + "/combinedGlossary.en"
        dictionaryB = root_dir + "/combinedGlossary.ta"
        # dictionaryA = root_dir + "/existingdictionary.en"
        # dictionaryB = root_dir + "/existingdictionary.ta"
        personNamesA = root_dir + "/person-names.en"
        personNamesB = root_dir + "/person-names.ta"
        designationsA = root_dir + "/designations.en"
        designationsB = root_dir + "/designations.ta"

    if language_pair == 'si-ta':
        dictionaryA = root_dir + "/combinedDictionaryNew.ta"
        dictionaryB = root_dir + "/combinedDictionaryNew.si"
        # dictionaryA = root_dir + "/existingdictionary.ta"
        # dictionaryB = root_dir + "/existingdictionary.si"
        personNamesA = root_dir + "/person-names.ta"
        personNamesB = root_dir + "/person-names.si"
        designationsA = root_dir + "/designations.ta"
        designationsB = root_dir + "/designations.si"

    # wordDictionary = []

    with open(dictionaryA) as dictionaryFileA:
        with open(dictionaryB) as dictionaryFileB:
            linesA = dictionaryFileA.readlines()
            linesB = dictionaryFileB.readlines()
            for i in range(len(linesA)):
                word = linesA[i].strip().replace("\n", "").lower()
                if (wordDictionary.get(word, False)):
                    if (linesB[i].strip().replace("\n", "") not in wordDictionary.get(word)):
                        wordDictionary[word].append(linesB[i].strip().replace("\n", ""))
                else:
                    wordDictionary[word]  = [linesB[i].strip().replace("\n", "")]

    with open(designationsA) as designationsFileA:
        with open(designationsB) as designationsFileB:
            linesA = designationsFileA.readlines()
            linesB = designationsFileB.readlines()
            for i in range(len(linesA)):
                word = linesA[i].strip().replace("\n", "").lower()
                if (wordDictionary.get(word, False)):
                    wordDictionary[word].append(linesB[i].strip().replace("\n", ""))
                else:
                    wordDictionary[word]  = [linesB[i].strip().replace("\n", "")]

    with open(personNamesA) as personNamesA:
        with open(personNamesB) as personNamesB:
            namesA = personNamesA.readlines()
            namesB = personNamesB.readlines()
            for  i in range(len(namesA)):
                nameA = namesA[i].strip().replace("\n", "")
                if (wordDictionary.get(nameA, False)):
                    wordDictionary[nameA].append(namesB[i].strip().replace("\n", ""))
                else:
                    wordDictionary[nameA] = [namesB[i].strip().replace("\n", "")]

    return wordDictionary


def load_sdml_model(language_pair):
    filename = '/media/laka/Lakmali/FYP/ml_paper/'+ language_pair + '/SDML.sav'
    sdml = pickle.load(open(filename, 'rb'))
    return sdml


def load_itml_model(language_pair):
    filename = '/media/laka/Lakmali/FYP/ml_paper/'+ language_pair + '/ITML.sav'
    itml = pickle.load(open(filename, 'rb'))
    return itml


if __name__ == "__main__":
    main()
