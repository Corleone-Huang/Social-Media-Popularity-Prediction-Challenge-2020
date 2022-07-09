import numpy as np
import csv
import pandas as pd
from document import Document
from alias import AliasSampler, SparseAliasSampler
from tqdm import trange
import argparse

class lightLDA(object):
    def __init__(self, K: int, docs: Document, num_MH=2) -> None:
        self.K = K
        self._documents = docs.get_documents()
        self._V = docs.get_num_vocab()
        self._D = docs.get_num_docs()
        self._beta = 0.1
        self._Vbeta = self._V * self._beta
        self._alpha = 0.01
        self._sum_alpha = 0.1 * K
        self._nkv = np.zeros((self.K, self._V)).astype(np.int32)
        self._ndk = np.zeros((self._D, self.K)).astype(np.int32)
        self._nk = np.zeros(self.K).astype(np.int32)
        self._z = []
        self.num_MH = num_MH

    def fit(self, num_iterations=300) -> None:
        # random init topic
        for doc_id, doc in enumerate(self._documents):
            doc_topic = np.random.randint(self.K, size=doc.shape[0], dtype=np.uint32)
            self._z.append(doc_topic)
            for word, topic in zip(doc, doc_topic):
                self._nkv[topic, word] += 1
                self._ndk[doc_id, topic] += 1
                self._nk[topic] += 1

        alpha_table = AliasSampler(p=np.ones(self.K))
        denominator_part_beta_nk_or_beta = self.K * self._Vbeta
        denominator_nk_or_beta = np.sum(self._nk) + denominator_part_beta_nk_or_beta

        for ite in trange(1, num_iterations + 1):
            # create alpha table
            word_proposal_denom = (self._nk + self._Vbeta)
            beta_talbe = AliasSampler(p=self._beta / word_proposal_denom)
            word_tables = []
            for v in range(self._V):
                topics = np.nonzero(self._nkv[:, v])[0]
                p = np.array([self._nkv[k, v] / word_proposal_denom[k] for k in topics])
                word_tables.append(SparseAliasSampler(p=p, topics=topics))

            for d in range(self._D):
                w_d = self._documents[d]
                N_d = w_d.shape[0]
                for i, w in enumerate(w_d):
                    old_topic = s = self._z[d][i]
                    for _ in range(self.num_MH):

                        # word proposal
                        nk_or_beta = np.random.rand() * denominator_nk_or_beta

                        if nk_or_beta < denominator_part_beta_nk_or_beta:
                            t = beta_talbe.sample()
                        else:
                            t = word_tables[w].sample()

                        if t != s:
                            nsw = self._nkv[s, w]
                            ntw = self._nkv[t, w]
                            ns = self._nk[s]
                            nt = self._nk[t]

                            nsd_alpha = self._ndk[d, s] + self._alpha
                            ntd_alpha = self._ndk[d, t] + self._alpha
                            nsw_beta = nsw + self._beta
                            ntw_beta = ntw + self._beta
                            ns_Vbeta = ns + self._Vbeta
                            nt_Vbeta = nt + self._Vbeta

                            proposal_nominator = nsw_beta * nt_Vbeta
                            proposal_denominator = ntw_beta * ns_Vbeta

                            if s == old_topic:
                                nsd_alpha -= 1.
                                nsw_beta -= 1.
                                ns_Vbeta -= 1.

                            if t == old_topic:
                                ntd_alpha -= 1.
                                ntw_beta -= 1.
                                nt_Vbeta -= 1.

                            pi_nominator = ntd_alpha * ntw_beta * ns_Vbeta * proposal_nominator
                            pi_denominator = nsd_alpha * nsw_beta * nt_Vbeta * proposal_denominator

                            pi = (pi_nominator / pi_denominator).item()

                            m = -(np.random.rand() < pi)
                            s = (t & m) | (s & ~m)

                        # doc proposal
                        nd_or_alpha = np.random.rand() * (N_d + self._sum_alpha)

                        if N_d > nd_or_alpha:
                            t = self._z[d][int(nd_or_alpha)]
                        else:
                            t = alpha_table.sample()

                        if t != s:
                            nsd = self._ndk[d, s]
                            ntd = self._ndk[d, t]

                            nsd_alpha = proposal_nominator = nsd + self._alpha
                            ntd_alpha = proposal_denominator = ntd + self._alpha
                            nsw_beta = self._nkv[s, w] + self._beta
                            ntw_beta = self._nkv[t, w] + self._beta
                            ns_Vbeta = self._nk[s] + self._Vbeta
                            nt_Vbeta = self._nk[t] + self._Vbeta

                            if s == old_topic:
                                nsd_alpha -= 1.
                                nsw_beta -= 1.
                                ns_Vbeta -= 1.

                            if t == old_topic:
                                ntd_alpha -= 1.
                                ntw_beta -= 1.
                                nt_Vbeta -= 1.

                            pi_nominator = ntd_alpha * ntw_beta * ns_Vbeta * proposal_nominator
                            pi_denominator = nsd_alpha * nsw_beta * nt_Vbeta * proposal_denominator

                            pi = (pi_nominator / pi_denominator).item()
                            m = -(np.random.rand() < pi)
                            s = (t & m) | (s & ~m)

                    # update topic
                    if s != old_topic:
                        self._z[d][i] = s

                        self._nkv[old_topic, w] -= 1
                        self._ndk[d, old_topic] -= 1
                        self._nk[old_topic] -= 1

                        self._nkv[s, w] += 1
                        self._ndk[d, s] += 1
                        self._nk[s] += 1

    def word_predict(self, topic: int) -> np.ndarray:
        return (self._nkv[topic, :] + self._beta) / (self._nk[topic] + self._Vbeta)

    def topic_predict(self, doc_id: int) -> np.ndarray:
        p = self._ndk[doc_id, :] + self._alpha
        return p / np.sum(p)


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="lightlda_tag")
    parser.add_argument('--num_of_topic', '-K', type=int)
    parser.add_argument('--num_of_iteration', '-IT', type=int)
    parser.add_argument('--input_processed_data_path', '-i', type=str,
                        help='the path of input data，pre_tag.txt')
    parser.add_argument('--output_topic_word_path', '-tw', type=str,
                        help='the path of topic_word.csv')
    parser.add_argument('--output_doc_topic_path', '-dt', type=str,
                        help='the path of document_topic.csv')
    parser.add_argument('--data_path', '-data', type=str,
                        help='the path of data，train_tags.json')
    args = parser.parse_args()

    np.random.seed(1112)
    print('# Start training...')
    K = args.num_of_topic
    docs = Document().fit(args.input_processed_data_path)
    model = lightLDA(K=K, docs=docs, num_MH=2)
    model.fit(num_iterations=args.num_of_iteration)

    print('# Word distributions per latent class')
    topic_word_list = []
    for k in range(K):
        print('## φ of latent class of {}'.format(k))
        print('word: probability')
        topic_word_list_2 = []
        d = {docs.get_word(i): p for i, p in enumerate(model.word_predict(topic=k))}
        for v, p in sorted(d.items(), key=lambda x: -x[1]):
            topic_word_list_2.append(v)
            print('{}: {:.3f}'.format(v, p))
        topic_word_list.append(topic_word_list_2)
        print()
# 存至文档
    with open(args.output_topic_word_path, 'w') as f:
        topic_word_list = np.array(topic_word_list)
        wr = csv.writer(f)
        wr.writerows(topic_word_list)


    print('# Topic distributions per document')
    theta_list = []
    for doc_id in range(docs.get_num_docs()):
        print('## Topic information of document {}'.format(doc_id))
        theta = model.topic_predict(doc_id=doc_id)
        theta_list.append(theta)
        topics = model._z[doc_id]
        print('Propotion of topics')
        print('topic: θ_{document_id, latent_class}')
        for k, p in enumerate(theta):
            print('{}: {:.3f}'.format(k, p))

        print('\nAssigned latent class per word')
        print('word: latent class')
        for w, z in zip(docs.get_document(doc_id), topics):
            print('{}: {} '.format(docs.get_word(w), z))
        print('--------------\n')

# 存至文档
    df_tags = pd.read_json(agrs.data_path)
    df = df_tags[["Uid", "Pid", "Title", "Alltags"]]
    uid = np.array(df["Uid"])
    pid = np.array(df["Pid"])
    uid = uid.reshape(uid.shape[0], 1)
    pid = pid.reshape(pid.shape[0], 1)
    with open(args.output_doc_topic_path, 'w') as f:
        theta_list = np.array(theta_list)
        wr = csv.writer(f)
        wr.writerow(['uid', 'pid', 'tag'])
        data = np.hstack((uid, pid))
        dt_save = np.hstack((data, theta_list))
        wr.writerows(dt_save)
