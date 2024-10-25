import torch


def epoch_class_recall(class_names, epoch, all_labels, all_predictions, history):
    """
    TODO
    """
    
    # Initializations
    all_labels = torch.cat(all_labels)
    all_predictions = torch.cat(all_predictions)
    recalls = []
    
    
    # Calculates the recall for each class
    for class_idx in range(len(class_names)):
        
        TP = ((all_predictions == class_idx) & (all_labels == class_idx)).sum().item()
        FN = ((all_predictions != class_idx) & (all_labels == class_idx)).sum().item()
        
        if (TP + FN) > 0:
            recall = TP / (TP + FN)
        else:
            recall = 0
        
        recalls.append(recall)
        
        print(f'Recall for class {class_idx}: {recall}')
    
    
    # Calculates the average recall for all classes (macro-average recall)
    macro_average_recall = sum(recalls) / len(recalls)
    
    print(f'Macro-average recall for epoch {epoch}: {macro_average_recall}')
    
    
    # Adds the recalls to the history
    recalls.append(macro_average_recall)
    history['recall'].append(recalls)
