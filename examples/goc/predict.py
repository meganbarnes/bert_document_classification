import sys, os, logging, torch

#appends current directory to sys path allowing data imports.
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append("/home/mrbarnes/projects/bert_document_classification")
from data import load_goc
from sklearn.metrics import f1_score, precision_score, recall_score
from bert_document_classification.models import GOCBert
import configargparse

log = logging.getLogger()

if __name__ == "__main__":

    test = list(load_goc(partition='TEST_SET'))

    print(len(test))

    smoking_bert_phenotyper = GOCBert() #CPU prediction (change to 'cuda' if possible)
    labels = smoking_bert_phenotyper.labels

    test_documents, test_labels = [],[]
    for _, text, status in test:
        test_documents.append(text)
        label = [0]*len(labels)
        for idx, name in enumerate(labels):
            if name == status:
                label[idx] = 1
        test_labels.append(label)

    correct_labels = torch.FloatTensor(test_labels).transpose(0,1)

    predictions = smoking_bert_phenotyper.predict(test_documents)
    assert correct_labels.shape == predictions.shape

    precisions = []
    recalls = []
    fmeasures = []
    for label_idx in range(predictions.shape[0]):
        correct = correct_labels[label_idx].view(-1).numpy()
        predicted = predictions[label_idx].view(-1).numpy()
        present_f1_score = f1_score(correct, predicted, average='binary', pos_label=1)
        present_precision_score = precision_score(correct, predicted, average='binary', pos_label=1)
        present_recall_score = recall_score(correct, predicted, average='binary', pos_label=1)

        precisions.append(present_precision_score)
        recalls.append(present_recall_score)
        fmeasures.append(present_f1_score)


    print('Metric\t' + '\t'.join([labels[label_idx] for label_idx in range(predictions.shape[0])]))
    print('Precision\t' + '\t'.join([str(precisions[label_idx]) for label_idx in range(predictions.shape[0])]))
    print('Recall\t' + '\t'.join([str(recalls[label_idx]) for label_idx in range(predictions.shape[0])]))
    print('F1\t' + '\t'.join([str(fmeasures[label_idx]) for label_idx in range(predictions.shape[0])]))
    print('Micro F1\t' + str(f1_score(correct_labels.reshape(-1).numpy(), predictions.reshape(-1).numpy(), average='micro')) )
    print('Macro F1\t' + str(f1_score(correct_labels.reshape(-1).numpy(), predictions.reshape(-1).numpy(), average='macro')) )
