# deeplearning_midterm_22300130110
The midterm project of deep learning, based on MNIST. The repository provided the codes, the weight of the models and the dataset that the project requested, including a mindspore version CNN(including a common CNN and the classical LeNet) model for slow implementation of the original CNN model in "test_train.py".
1. In the "dataset" fold, there are the MNIST set(same as the ones that downloaded from the Github link of elearning).
2. In the "saved_models" fold, there contains the weights and accuracy of different models in the "modification_model.pickle" and "modification_model_history.pickle" respectively. The modifications were accumulated according to the questions' sequence of the pj request in 1.2 questions.
3. The weight and the accuracy of the models are visiualized in the "figs" fold, including the weight visualization of the layers, accuracy and loss curve. The images, if not denoted like "dropout_MLP_withoutl2", accumulate all the changes as the sequence of the 1.2 questions of the project.
4. You can adjust and restore the experiment through modify the "test_train.py" or "mindspore_CNN.py"
