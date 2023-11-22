import os
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

text_list = []
corpus = []
with open(os.path.join('/data/','candidate_all.tsv')) as file:
    for line in file:
        text_list.append(line.strip())

with open('candidate_subset.tsv', "w", encoding="utf-8") as outf:
    for i in tqdm(range(int(len(text_list)/2003))):
        corpus = []

        #query
        corpus.append(text_list[1+i*2003][6:])
        person = text_list[i*2003][8:]
        gold = text_list[2+i*2003][5:]

        corpus = corpus+text_list[3+i*2003:3+i*2003+2000] #add candidate

        tfidf_vec = TfidfVectorizer()
        tfidf_matrix = tfidf_vec.fit_transform(corpus)

        vec = tfidf_matrix.toarray()

        sim = []
        for j in range(1,2001):
            sim.append(cosine_similarity(vec[0].reshape(1, -1), vec[j].reshape(1, -1)))

        sim_list = []
        for j in range(2000):
            sim_list.append(sim[j][0][0])

        text_sim = []
        for j in range(2000):
            text_sim.append([corpus[j+1],sim_list[j]])

        text_sim = sorted(text_sim,key=lambda x:x[1], reverse=True)

        outf.write(f"persona:{person}\n")
        outf.write(f"query:{corpus[0]}\n")
        outf.write(f"gold:{gold}\n")
        for j in range(100):
            outf.write(f"{text_sim[j][0]}\n")