# <b>Logging and Checkpointing</b>
This module contains two submodules: Logging and Checkpointing. The logging module helps the training process by logging the current progress to a file. The checkpointing module helps the training process by saving a training checkpoint every certian number of epochs. 

--- Chechpointing uses logging ?

<hr> <hr>

# <b>Logging</b>: <i>logger.py</i>
<p>
    The log file(.log) is stored as a pickle file. It is a dictionary of items ( objects, boolean, etc). Logging is done both for sequential and distributed training. It's purpose is to keep track of what to train, as well as its current progress. The following items are stored at some point in the program:
</p>

1) "test_subjects" or "is_finished"
2) "current_rotation"
3) "fold_info"

<hr>

## <b>1.a) Test subjects</b>: <i>run_training.py</i>
The list of strings that contains test subjects that have yet to train are stored from within the main function. 

--- Plusieurs sujets ? Qu'est-ce qu'un sujet ?

## <b>1.b) Is Finished</b>: <i>mpi_processing.py</i>
A boolean is stored for each subject or subject-pair that is finished. This is set after training is complete. If it is revisited, the worker can immedietly ignore it.

--- Des paires de sujets ?

## <b>2) Rotation</b>: <i>training_loop.py</i>
The current rotation is stored in the log within the training loop. For jobs with only one subject or subject-pair, this will be 1. Else 1-n. If the current rotation were 0, it will have no value. The rotation is stored as a dict. <i>{test_subject: rotation+1}</i>

--- Qu'est-ce qu'une rotation ? Un passage dans la boucle ?

## <b>3) Fold Information</b>: <i>training_fold.py</i>
Within the training fold, the fold information is stored into the log. The fold information is it's own class, <i>_FoldTrainingInfo</i>, and contains many items. These are things like the dataset information, subjects, rank, configuration, etc. The dataset cannot be saved, due to some limitation with pickling TensorFlow tensors. It must be recreated after the log is read in.

--- Qu'est-ce qu'un fold ?

<br> <br> <hr> <hr>

# <b>Checkpoints</b>: <i>checkpointer.py</i>
<p> 
Checkpoints are models that are periodically saved during the training process on some <i>n</i> epochs. They are loaded in whenever a training job is stopped and resumed. They are not stored or associated with the log. They are created as callbacks within the fold information and specified within the training function. They are identified by their training subject(s) and their greatest epoch count. A customized checkpoint class was created for this purpose.
</p>

--- Checkpoint = loggin automatique ?