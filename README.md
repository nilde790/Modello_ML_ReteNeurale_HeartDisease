Machine Learning Project for Heart Disease
Questo progetto analizza un dataset clinico relativo al rischio di malattie cardiache e confronta due modelli di classificazione binaria:

Regressione Logistica
Rete Neurale sviluppata con PyTorch
L'obiettivo e' prevedere la variabile target HeartDisease, dove:

0 indica assenza di malattia cardiaca
1 indica presenza di malattia cardiaca
Dataset
Il dataset utilizzato contiene 918 osservazioni e 12 colonne iniziali:

Age
Sex
ChestPainType
RestingBP
Cholesterol
FastingBS
RestingECG
MaxHR
ExerciseAngina
Oldpeak
ST_Slope
HeartDisease
Durante l'analisi sono stati individuati valori mancanti in alcune colonne, in particolare:

RestingBP: 1 valore mancante
Cholesterol: 172 valori mancanti
I valori mancanti sono stati gestiti tramite imputazione con la mediana, scelta adatta soprattutto per Cholesterol, dove sono presenti diversi outlier.

Analisi e Preprocessing
Il notebook segue questi passaggi principali:

Caricamento del dataset con Pandas
Analisi esplorativa iniziale tramite head, info e describe
Controllo dei valori nulli
Imputazione dei valori mancanti con la mediana
Analisi delle variabili categoriche
Codifica delle variabili categoriche tramite One-Hot Encoding
Divisione del dataset in training set e test set
Standardizzazione delle feature numeriche con StandardScaler
Addestramento e valutazione dei modelli
Le variabili categoriche codificate sono:

Sex
ChestPainType
RestingECG
ExerciseAngina
ST_Slope
Dopo il One-Hot Encoding, il dataset passa da 12 a 16 colonne.

Modelli Utilizzati
Regressione Logistica
La Regressione Logistica e' stata scelta come primo modello perche' e' adatta ai problemi di classificazione binaria, e' semplice da interpretare e funziona bene anche con dataset di dimensioni non molto grandi.

Il modello e' stato addestrato con:

LogisticRegression
solver='liblinear'
max_iter=1000
random_state=67
Rete Neurale con PyTorch
Il secondo modello e' una rete neurale semplice realizzata con PyTorch. La rete contiene:

un layer di input
un layer nascosto con 16 neuroni
funzione di attivazione ReLU
Dropout con valore 0.3
un layer di output con funzione Sigmoid
La rete e' stata addestrata per 250 epoche usando:

BCELoss come funzione di perdita
Adam come ottimizzatore
learning rate pari a 0.001
Risultati
Regressione Logistica
La Regressione Logistica ha ottenuto:

Accuracy: 0.8913
Macro Avg Precision: 0.89
Macro Avg Recall: 0.89
Macro Avg F1-score: 0.89
Matrice di confusione:

[[71 11]
 [ 9 93]]
Rete Neurale
La rete neurale PyTorch ha ottenuto circa:

Accuracy: 0.875
Macro Avg Precision: 0.87
Macro Avg Recall: 0.87
Macro Avg F1-score: 0.87
Matrice di confusione:

[[71 11]
 [12 90]]
Confronto dei Modelli
Nel confronto finale, la Regressione Logistica risulta leggermente migliore della rete neurale. La differenza non e' enorme, ma e' significativa perche' il contesto medico richiede attenzione soprattutto alla riduzione dei falsi negativi.

La Regressione Logistica ha mostrato:

migliori metriche complessive
maggiore semplicita'
migliore interpretabilita'
minore rischio di overfitting su un dataset relativamente piccolo
La rete neurale, invece, pur essendo piu' flessibile, non ha portato un miglioramento concreto. Questo puo' dipendere dalla dimensione limitata del dataset e dal fatto che le relazioni tra le feature e la variabile target non sembrano richiedere un modello molto complesso.

Tecnologie Utilizzate
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
PyTorch
Jupyter Notebook
Come Eseguire il Progetto
Clonare il repository:
git clone <url-del-repository>
Entrare nella cartella del progetto:
cd <nome-repository>
Installare le dipendenze principali:
pip install pandas numpy matplotlib seaborn scikit-learn torch
Avviare Jupyter Notebook:
jupyter notebook
Aprire il file:
Machine learning project for heart disease.ipynb
Conclusione
Per questo progetto di previsione delle malattie cardiache, la Regressione Logistica si e' dimostrata il modello piu' efficace. Nonostante la rete neurale sia uno strumento potente, in questo caso il modello piu' semplice ha ottenuto risultati migliori ed e' anche piu' facile da interpretare.

Questo rende la Regressione Logistica una scelta adatta per un primo approccio predittivo su questo dataset.
