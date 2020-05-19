import sqlite3

def load_goc(partition='TRAINING_SET'):
    """
    Yields a generator of id, doc, label tuples.
    :param partition:
    :return:
    """

    ids = []
    notes = []
    labels = []

    conn = sqlite3.connect("/usr/local/share/cambia_nlp/merged_splits.db")
    curs = conn.cursor()
    notes = curs.execute("SELECT * FROM %s" % partition).fetchall()
    conn.close

    for document in notes:
        ids.append(document[0])
        notes.append(document[1])
        labels.append(document[2])

    for id_, note, label in zip(ids,notes,labels):
        yield (id_,note,label)