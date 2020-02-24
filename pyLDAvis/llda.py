"""
pyLDAvis llda
===============
Helper functions to visualize Labeled LDA models
"""

import funcy as fp
import pyLDAbis.pyLDAbis as plb


def _get_doc_lengths(dtm):
    return dtm.sum(axis=1).getA1()


def _get_term_freqs(dtm):
    return dtm.sum(axis=0).getA1()


def _extract_data(lda_model, dtm, labels):
    vocab = lda_model.vocas
    doc_lengths = _get_doc_lengths(dtm)
    term_freqs = _get_term_freqs(dtm)
    topic_term_dists = lda_model.phi()
    err_msg = ('Topic-term distributions and document-term matrix'
               'have different number of columns, {} != {}.')

    assert term_freqs.shape[0] == len(vocab), \
        ('Term frequencies and vocabulary are of different sizes, {} != {}.'
         .format(term_freqs.shape[0], len(vocab)))

    assert topic_term_dists.shape[1] == dtm.shape[1], \
        (err_msg.format(topic_term_dists.shape[1], len(vocab)))

    # column dimensions of document-term matrix and topic-term distributions
    # must match first before transforming to document-topic distributions
    doc_topic_dists = lda_model.theta()
    return {'vocab': vocab,
            'doc_lengths': doc_lengths.tolist(),
            'term_frequency': term_freqs.tolist(),
            'doc_topic_dists': doc_topic_dists.tolist(),
            'topic_term_dists': topic_term_dists.tolist(),
            'labels': labels}


def prepare(lda_model, dtm, labels, **kwargs):
    """Create Prepared Data from sklearn's LatentDirichletAllocation and CountVectorizer.

    Parameters
    ----------
    lda_model : sklearn.decomposition.LatentDirichletAllocation.
        Latent Dirichlet Allocation model from sklearn fitted with `dtm`

    dtm : array-like or sparse matrix, shape=(n_samples, n_features)
        Document-term matrix used to fit on LatentDirichletAllocation model (`lda_model`)

    vectorizer : sklearn.feature_extraction.text.(CountVectorizer, TfIdfVectorizer).
        vectorizer used to convert raw documents to document-term matrix (`dtm`)

    **kwargs: Keyword argument to be passed to pyLDAvis.prepare()


    Returns
    -------
    prepared_data : PreparedData
          the data structures used in the visualization


    Example
    --------
    For example usage please see this notebook:
    http://nbviewer.ipython.org/github/bmabey/pyLDAvis/blob/master/notebooks/sklearn.ipynb

    See
    ------
    See `pyLDAvis.prepare` for **kwargs.
    """
    opts = fp.merge(_extract_data(lda_model, dtm, labels), kwargs)
    return plb.prepare(**opts)
