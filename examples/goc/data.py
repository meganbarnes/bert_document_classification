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
    #conn = sqlite3.connect("/home/mrbarnes/fake_merged_splits")
    curs = conn.cursor()
    notes = curs.execute("SELECT * FROM %s" % partition).fetchall()
    conn.close

    print([x[2] for x in notes])

    for item in notes:
        yield(item[:3])

