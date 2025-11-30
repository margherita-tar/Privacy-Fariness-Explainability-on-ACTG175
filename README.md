# AIDS-clinical-trials
**1. INTRODUZIONE**
1.1 Spiegazione del progetto e obiettivo del lavoro
Il Dataset “AIDS Clinical Trials Group Study 175”, originariamente pubblicato nel 1996, rappresenta una raccolta dettagliata di dati clinici e informazioni categoriali relative a pazienti diagnosticati con AIDS. Questo dataset è stato concepito principalmente per valutare l’efficacia di diverse strategie terapeutiche per il trattamento della malattia, in particolare considerando tecniche di monoterapia e terapia combinata.

L'obiettivo del lavoro è quello di implementare tecniche di tutela della privacy, di verifica dell'equità e di analisi di spiegabilità, valutando le prestazioni del modello mediante Regressione Logistica.

Il focus del progetto verte in particolare su:

Fairness: Demographic Parity (DP), Equal Opportunity (EOD), treshold shifting
Interpretability & Explainability: SHAP & PDP
Anonimizzazione: K-anonimity
1.2 Descrizione del Dataset
Il dataset è composto da 2139 samples e 23 features + target label. La previsione di sopravvivenza/morte del paziente si basa su:

Caratteristiche generali dell'individuo (dati anagrafici e comportamentali)
Informazioni sul trattamento e storia clinica
Stato di salute e valori immunologici

**2. SET-UP**
Import delle librerie
Import del Dataset come file CSV e visualizzazione

**3. ANALISI DEI DATI (EDA & PLOTS)**
Valutando la distribuzione delle classi nella variabile target per verificare un eventuale sbilanciamento nei dati, è emerso che la classe predominante risulta essere label = 0 (paziente sopravvissuto).

Dalla descrizione del dataset emerge che alcune features sono ridondanti rispetto al nostro obiettivo, dunque per rendere più agevole l'analisi predittiva alcune di esse sono state rimosse. Ad esempio, riguardo la storia antiretrovirale dei pazienti, si è deciso di mantenere le features più informative rispetto alle loro corrispettive generalizzazioni: strat invece di str2, cd40 e cd80 rispetto a cd420 e cd820, z30 invece di zprior e trt al posto di treat.

EDA: per avere un EDA più mirato distinguiamo tra variabili numeriche e categoriche

shape: 2139 righe e 19 colonne
dtypes: tipologia di dato
isna: valori mancanti
describe: statistiche descrittive interessanti (media, deviazione standard e quartili)
Correlazione tra features

Correlation matrix con heatmap (non risultano correlazioni significative)
Pairplot
Istogrammi per visualizzare la distribuzione dei valori di label per le features categoriche sia binarie che nominali. Notiamo che il trend è uno sbilanciamento verso la classe 0, come già suggerito dalla label target

**4. PRE-PROCESSING E FEATURE ENGINEERING**
Encoding sulle feature nominali: trt & strat
Separazione della variabile target dalle features
Separazione in training e test set
Standardizzazione dei dati

**5. LOGISTIC REGRESSION (modello di base e miglior modello)**
Usiamo un modello semplice di Regressione Logistica per valutare le prestazioni di un modello di base con le metriche standard di:

Precision
Recall
F1-score
Accuracy
Per rendere l'analisi più accurata proviamo a migliorare il modello correggendo lo sbilanciamento tra classi tramite l'utilizzo del parametro class_weight = 'balanced' e implementazione di Grid Search con ricerca di iperparametri per Cross-validation.

Paragonando i due modelli notiamo che:

Accuratezza del modello base: 0.836
Accuratezza del miglior modello: 0.859
Inoltre, al fine di ottenere ulteriori informazioni in merito all'apprendimento del modello sono state plottate le curve di precision-recall e ROC-AUC, ottenendo un valore di AUC di 0.887.

Infine, per escludere la possibilità di overfitting è stata plottata una curva di apprendimento per visualizzare l'andamento delle performance del miglior modello sui dati di training e di test. Si evince che il modello generalizza bene sui dati:

Accuratezza training set: 0.8551
Accuratezza test set: 0.8598

**6. FAIRNESS-EQUITA'**
Si analizza quanto il modello tratta equamente i dati, relativamente a tre attributi sensibili della persona, che hanno più possibilità di essere oggetto di bias discriminativi:

'homo'
'gender'
'race'
E' stata utilizzata una funzione analyze_fairness che valuta metriche di fairness di gruppo, calcolando:

Demographic Parity (DP) → tasso di selezione: probabilità che un individuo sia classificato come positivo
Equal Opportunity (EO) → tasso di veri positivi: probabilità che un individuo vero positivo sia effettivamente classificato come positivo
Per comprendere e visualizzare se esiste una disparità di trattamento, la funzione stampa per ogni attributo sensibile, per entrambi i valori (0/1):

Classification Report (per individuare veri e falsi positivi)
DPD: differenza di parità demografica tra i due gruppi
EOD: differenza di uguaglianza di opportunità tra i due gruppi
Grafico PD
Grafico EO
- Demographic Parity Difference (DPD)

DPD=P(Y^=1∣A=0)−P(Y^=1∣A=1) 

Dove:

Y^  è la predizione del modello
A  è l’attributo protetto (es. genere, etnia)
- Equal Opportunity Difference (EOD)

EOD=P(Y^=1∣Y=1,A=0)−P(Y^=1∣Y=1,A=1)

Dove:

Y=1  rappresenta la classe positiva reale
DPD = 0 e EOD = 0 sono i risultati ideali di equità.

Osservazioni:
Dai risultati stampati possiamo notare che la feature 'homo' contiene la disparità più marcata:

DPD: 0.0959: differenza di 9,6 punti percentuali tra i due gruppi, dunque una disequità non grave;
EOD = 0.2610: differenza di 26,1 punti percentuali tra i due gruppi, risultato più preoccupante.
Poichè EOD per l'attributo 'homo' è abbastanza alta, si procede con un tentativo di mitigazione del bias.

Mitigazione del bias
Poichè il problema di equità principale risiede nell'EOD, si tenta di rendere più equo il modello modificando la soglia decisionale (threshold) per il gruppo svantaggiato (homo=0), in modo che il TPR sia simile a quello del gruppo privilegiato (homo=1).

Si esplorano soglie da 0.01 a 0.99 per trovare quella che rende il TPR del gruppo homo=0 più vicino a quello di homo=1, dunque la si applica solo per homo=0. Per entrambi i gruppi vengono ricalcolati:

metriche principali di accuracy, recall e precision
DPD
EOD
Osservazioni:
Metrica	Prima Mitigazione	Dopo Mitigazione
TPR (homo=1)	0.847	0.847 (target mantenuto)
TPR (homo=0)	0.586	0.828
EOD (TPR diff.)	0.261	0.020 (migliorata)
Tasso di Selezione (homo=1)	0.301	0.301 (invariata)
Tasso di Selezione (homo=0)	0.205	0.411
DPD (Demographic Parity Diff.)	0.096	-0.110
Equal Opportunity migliorata: è passato da 0.261 a 0.020, evidenziando un miglioramento nell’equità nel riconoscere i veri positivi per homo=0.
Demographic Parity compensata: è cambiato da +0.096 a -0.110. Sebbene ora il gruppo homo=0 abbia un tasso di selezione più alto, il bias si è “invertito”, dovuto al fatto che ora il modello predice in generale più positivi per homo=0.
Recall del gruppo svantaggiato (homo=0): è aumentata da 0.586 a 0.828
Precisione e accuracy leggermente penalizzate per homo=0: trade-off accettabile per migliorare l’equità complessiva.

**7. EXPLAINABILITY (SHAP - PDP)**
Installazione e import di SHAP
Import di PartialDependenceDisplay da scikit-learn per visualizzare il Partial Dependence Plot
Si è proceduto nell'implementazione di due tecniche per la spiegabilità:

SHAP (SHapley Additive exPlanations):
Asse Y: il plot mostra le feature progressivamente più impattanti
Asse X: valori di SHAP per ogni sample del dataset per indicare quanto una determinata feature ha "spinto" la predizione verso una classe o l'altra. I valori positivi spingono la predizione verso la classe 1 e viceversa. Sample contrassegnati in rosso rappresentano un valore alto per quella feature.
Ad esempio, time:

feature più impattante
esempi contrassegnati in blu (valori bassi della feature) sono associati a valori alti di SHAP (tendenza verso classe 1)
esempi contrassegnati in rosso (valori alti della feature) sono associati a valori bassi di SHAP (tendenza verso classe 0).
Ciò ha senso perchè per valori bassi di time c'è più probabilità che il paziente sia deceduto o abbia meno speranza di sopravvivere e viceversa.

PDP (Partial Dependence Plot):
dopo aver constatato quali features risultano più impattanti, si è deciso di plottare PDP per 3 features significative, ma appartenenti a categorie differenti:

time: a rappresentare la storia clinica del paziente
cd40: a rappresentare un valore immunologico (biomarcatore che riflette lo stato di salute del paziente e la risposta immunitaria)
age: a rappresentare un dato personale dell'individuo
Infine è stato plottato un grafico 3D per mostrare la relazione che sussiste tra le features selezionate per esplorare come la dipendenza parziale della predizione cambia con la durata del trattamento e il livello di cd40, per differenti fasce d’età (slice su age in quanto l'età è una caratteristica fissa del paziente).

**8. ANONIMIZZAZIONE (K-ANONIMITY & REGRESSIONE LOGISTICA)**
Applichiamo la tecnica del k-anonimato, che garantisce che ogni istanza del dataset sia indistinguibile da almeno altri k-1 rispetto a un insieme di quasi-identificatori (QID), ovvero attributi che, se combinati, possono rivelare l’identità di un individuo.

Obiettivo: soddisfare una soglia minima di k, attraverso una generalizzazione progressiva dei dati.

Abbiamo identificato come potenziali QID alcune variabili che rappresentano dati demografici e clinici:

age: età del paziente
wtkg: peso del paziente in kg
karnof: indice di Karnofsky (stato funzionale)
race: colore della pelle
gender: genere
Abbiamo definito una funzione min_k(), che calcola la dimensione minima dei gruppi di equivalenza, cioè il valore attuale di k per l'insieme di QID iniziali.

Abbiamo generalizzato alcune variabili numeriche per ridurre la granularità dell’informazione e quindi aumentare le dimensioni dei gruppi di equivalenza, in particolare dividendo in due gruppi le features:

age: in "<50" e ">=50"
wtkg: in "light" (<75kg) e "heavy" (≥75kg)
karnof: in "low" (0–50) e "high" (51–100)
Dopo la generalizzazione, applichiamo questa funzione al dataset per stimare il livello iniziale di anonimato.

Se il livello iniziale di k è inferiore alla soglia scelta (k_threshold=10), allora si procede con la generalizzazione di altri QID; visualizzando le distribuzioni di gender e race mediante istogrammi, si nota che la variabile gender è più sbilanciata, pertanto è più rilevante generalizzarla rispetto a race, poichè potrebbe essere più rappresentativa ed essere utilizzata come mezzo per risalire ai dati sensibili che, invece, vorremmo proteggere.

Dunque se gender è tra i QID, e k è troppo basso, lo si generalizza completamente sostituendo tutti i valori con "*", in modo da eliminare l'informazione completamente.

Output:

Stampa delle prime 5 righe del dataset anonimizzato

Analisi delle dimensioni dei gruppi di equivalenza:

16 gruppi di equivalenza
il gruppo più popoloso di record ne contiene 716, dunque un buon risultato, poichè un individuo che appartiene a quel gruppo è indistinguibile da tutti gli altri 715
Poichè abbiamo inserito forte granularità, è temibile una perdita di informazione con relativo peggioramento della performance del modello di Regressione Logistica applicato al nuovo dataset anonimizzato.

Per valutare se c'è giusto compromesso tra granularità e quantità di informazione, procederemo con una Logistic Regression basata sul nuovo dataset.

**9. CONFRONTO MODELLO di LOGISTIC REGRESSION: PRIMA vs DOPO ANONIMIZZAZIONE**
Strutturando il modello di Regressione Logistica come fatto con dataset non-anonimizzato, possiamo paragonare i risultati:

Il miglior parametro C è cambiato leggermente dal modello iniziale (0.1 → 1), segno che la distribuzione dei dati è leggermente cambiata, ma non in modo drastico.
Le metriche sono quasi identiche. C'è una leggera perdita di performance del modello su dataset anonimizzato (es. AUC da 0.8869 → 0.8822), prevedibile dopo aver utilizzato k-anonimity, poichè abbiamo un dataset con informazioni più generali.
Il recall della classe 1 è addirittura leggermente migliorato dopo l'anonimizzazione. In generale, il leggero miglioramento di alcune metriche potrebbe essere dato dalla perdita di rumore nel dataset grazie alla generalizzazione delle features che rappresentavano i QID.
Overfitting assente: in entrambi i casi: training e test sono molto vicini.
Conclusione
La k-anonimity ha avuto un impatto minimo, non ha degradato le prestazioni del modello.
Dunque si è raggiunto un giusto trade-off tra privatizzazione dei dati e conservazione di informazione.

**10. EXPLAINABILITY POST ANONIMIZZAZIONE (SHAP)**
Ripetere la tecnica SHAP di un modello a seguito di k-anonimity può essere utile per comprendere come le feature cambiano a livello di impatto sulle predizioni.

**11. CONSIDERAZIONI FINALI**
Giunte al termine del nostro lavoro abbiamo raccolto delle considerazioni in merito a motivazioni, aspettative e risultati che hanno guidato le nostre scelte nello svolgimento del progetto.

Innanzitutto, relativamente alla selezione del dataset AIDS Clinical Trials Group 175, abbiamo ritenuto che potesse soddisfare la nostra esigenza di focalizzarci su tecniche di privacy, spiegabilità ed equità, essendo un dataset clinico reale e quindi contenente dati sensibili. Tuttavia, per tale motivo vi è una complessità maggiore e presenza di rumore rispetto a proposte didattiche: le nostre aspettative iniziali intorno ai risultati che pensavamo potessero emergere non sempre sono state soddisfatte e perciò hanno ridirezionato il nostro lavoro.

Ad esempio contro le nostre previsioni, la performance del modello di Regressione Logistica applicato su dataset k anonimizzato non risente della significativa semplificazione, mostrando capacità predittive paragonabili al modello originario. Risultati confermati dalla stessa ripetizione della tecnica SHAP post anonimizzazione che rivela solo leggere differenze riguardo l'impatto delle features rispetto ai valori iniziali. Ciò è probabilmente dovuto al fatto che la variabilità causata dai numerosi record e features altamente specifiche rendono il dataset rumoroso e averlo generalizzato ha permesso di trovare importanti regolarità.

Questi esiti sono stati frutto di diversi tentativi guidati da valori differenti di k, generalizzazione di alcune feature piuttosto che altre e consultazione del paper "A TRIAL COMPARING NUCLEOSIDE MONOTHERAPY WITH COMBINATION THERAPY IN HIV-INFECTED ADULTS WITH CD4 CELL COUNTS FROM 200 TO 500 PER CUBIC MILLIMETER (The New England Journal of Medicine, Volume 335-Number 15).

Tentativi che non hanno riguardato solamente lo specifico caso dell'anonimizzazione, ma hanno indirizzato le decisioni relative anche alle altre strategie e argomenti di interesse: la scelta stessa del modello di Regressione Logistica, tecnica standard di classificazione, è risultata preferibile per raggiungere l'obiettivo di trade-off tra semplicità e buone prestazioni. A ciò si aggiunge l'idea per cui il focus del nostro progetto non era tanto raggiungere la migliore performance, ma sperimentare nuove tecniche di manipolazione del dataset e spiegabilità.

Per questo, una fase fondamentale è stata l'analisi del dataset originale che ci ha permesso di individuare delle macrocategorie su cui concentrarci a dispetto delle specifiche e a volte ridondanti features.

Non è certamente esclusa l'esplorazione di tecniche più avanzate e performanti che possano avere un impatto reale e significativo in ambito di salute pubblica, contesto ottimale per lo sviluppo e l'applicazione di varie strategie di protezione dati all'avanguardia, come il Federated Learning.
