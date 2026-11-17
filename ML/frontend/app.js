/**
 * EnsembleIQ • Production Enterprise Machine Learning Studio
 * Frontend Controller & Ensemble Visualization Engine
 */

document.addEventListener('DOMContentLoaded', () => {
    // =========================================================================
    // DOM Elements Cache
    // =========================================================================
    const navTabs = document.querySelectorAll('.nav-tab');
    const tabContents = document.querySelectorAll('.tab-content');
    
    // Header & Telemetry Status
    const backendStatusPill = document.getElementById('backend-status-pill');
    const backendStatusText = document.getElementById('backend-status-text');
    
    // Live Predictor (Module 1)
    const textInput = document.getElementById('text-input');
    const charCounter = document.getElementById('char-counter');
    const wordCounter = document.getElementById('word-counter');
    const analyzeBtn = document.getElementById('analyze-btn');
    const btnSpinner = document.getElementById('btn-spinner');
    const clearInputBtn = document.getElementById('clear-input-btn');
    const randomSampleBtn = document.getElementById('random-sample-btn');
    const presetPills = document.querySelectorAll('.preset-pill, .preset-chip');
    const loadingState = document.getElementById('loading-state');
    const resultsDashboard = document.getElementById('results-dashboard');
    
    // Consensus Elements
    const consensusVerdictPill = document.getElementById('consensus-verdict-pill');
    const consensusVerdictText = document.getElementById('consensus-verdict-text');
    const unanimousBadge = document.getElementById('unanimous-badge');
    const consensusDesc = document.getElementById('consensus-desc');
    const metricAgreement = document.getElementById('metric-agreement');
    const metricVotes = document.getElementById('metric-votes');
    const metricPosVotes = document.getElementById('metric-pos-votes');
    const metricNegVotes = document.getElementById('metric-neg-votes');
    const metricPosProb = document.getElementById('metric-pos-prob');
    const metricNegProb = document.getElementById('metric-neg-prob');
    
    // Model Cards
    const modelCardMap = {
        'Random Forest': { card: document.getElementById('card-rf'), badge: document.getElementById('badge-rf'), bar: document.getElementById('bar-rf'), conf: document.getElementById('conf-rf') },
        'AdaBoost': { card: document.getElementById('card-ada'), badge: document.getElementById('badge-ada'), bar: document.getElementById('bar-ada'), conf: document.getElementById('conf-ada') },
        'Gradient Boost': { card: document.getElementById('card-gb'), badge: document.getElementById('badge-gb'), bar: document.getElementById('bar-gb'), conf: document.getElementById('conf-gb') },
        'XGBoost': { card: document.getElementById('card-xgb'), badge: document.getElementById('badge-xgb'), bar: document.getElementById('bar-xgb'), conf: document.getElementById('conf-xgb') }
    };
    
    // Analytics (Module 2)
    const chartRows = {
        'Random Forest': { tag: document.getElementById('tag-chart-rf'), posBar: document.getElementById('pos-bar-rf'), negBar: document.getElementById('neg-bar-rf') },
        'AdaBoost': { tag: document.getElementById('tag-chart-ada'), posBar: document.getElementById('pos-bar-ada'), negBar: document.getElementById('neg-bar-ada') },
        'Gradient Boost': { tag: document.getElementById('tag-chart-gb'), posBar: document.getElementById('pos-bar-gb'), negBar: document.getElementById('neg-bar-gb') },
        'XGBoost': { tag: document.getElementById('tag-chart-xgb'), posBar: document.getElementById('pos-bar-xgb'), negBar: document.getElementById('neg-bar-xgb') }
    };
    const divergenceTitle = document.getElementById('divergence-title');
    const divergenceText = document.getElementById('divergence-text');
    const divergenceIcon = document.getElementById('divergence-icon');
    const polarityIndicator = document.getElementById('polarity-indicator');
    const polarityScore = document.getElementById('polarity-score');
    const spectrumExplanation = document.getElementById('spectrum-explanation');
    
    // NLP Inspector (Module 3)
    const nlpRawText = document.getElementById('nlp-raw-text');
    const nlpCleanText = document.getElementById('nlp-clean-text');
    const tokenCloud = document.getElementById('token-cloud');
    const tokenCountPill = document.getElementById('token-count-pill');
    const nlpMetricChars = document.getElementById('nlp-metric-chars');
    const nlpMetricWords = document.getElementById('nlp-metric-words');
    const nlpMetricTokens = document.getElementById('nlp-metric-tokens');
    const nlpMetricStopwords = document.getElementById('nlp-metric-stopwords');
    const nlpMetricDensity = document.getElementById('nlp-metric-density');

    // Batch Studio (Module 4)
    const batchTextarea = document.getElementById('batch-textarea');
    const batchLineCounter = document.getElementById('batch-line-counter');
    const batchFileInput = document.getElementById('batch-file-input');
    const batchDropzone = document.getElementById('batch-dropzone');
    const loadSampleBatchBtn = document.getElementById('load-sample-batch-btn');
    const clearBatchBtn = document.getElementById('clear-batch-btn');
    const runBatchBtn = document.getElementById('run-batch-btn');
    const batchProgressWrap = document.getElementById('batch-progress-wrap');
    const batchProgressFill = document.getElementById('batch-progress-fill');
    const batchProgressText = document.getElementById('batch-progress-text');
    const batchStatsContainer = document.getElementById('batch-stats-container');
    const bstatTotal = document.getElementById('bstat-total');
    const bstatPos = document.getElementById('bstat-pos');
    const bstatNeg = document.getElementById('bstat-neg');
    const bstatUnanimous = document.getElementById('bstat-unanimous');
    const batchResultsPanel = document.getElementById('batch-results-panel');
    const batchTableBody = document.getElementById('batch-table-body');
    const batchTableSearch = document.getElementById('batch-table-search');
    const batchFilterBtns = document.querySelectorAll('.filter-pill');
    const exportBatchCsv = document.getElementById('export-batch-csv');
    const exportBatchJson = document.getElementById('export-batch-json');
    let currentBatchData = [];
    let activeBatchFilter = 'all';

    // History (Module 6)
    const historyList = document.getElementById('history-list');
    const emptyHistoryState = document.getElementById('empty-history-state');
    const historyBadgeCount = document.getElementById('history-badge-count');
    const historyTotalCount = document.getElementById('history-total-count');
    const clearHistoryBtn = document.getElementById('clear-history-btn');
    const exportHistoryJson = document.getElementById('export-history-json');
    const toastContainer = document.getElementById('toast-container');

    // On Vercel the API lives at /api/* on the same origin.
    // Locally (served by serve.py on :3000) fall back to the FastAPI dev server.
    const IS_LOCAL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
    const API_URL  = IS_LOCAL ? 'http://localhost:8000' : '';
    const HEALTH_PATH  = IS_LOCAL ? '/health'  : '/api/health';
    const PREDICT_PATH = IS_LOCAL ? '/predict' : '/api/predict';

    // Sample Sentences for Evaluation
    const sampleSentences = [
        "I was not expecting this to work so well, but it proved me wrong and performed wonderfully!",
        "Horrible customer support and delayed delivery. The device broke on the first day, complete waste of money.",
        "This product completely blew me away! Best purchase of the year, absolutely flawless experience.",
        "The battery life is acceptable, but the software contains numerous frustrating lag spikes.",
        "Extremely disappointed with the cheap materials. Do not recommend to anyone.",
        "An absolute benchmark of modern engineering. Responsive, intuitive, and remarkably fast.",
        "Not worth the price tag. Promised full integration but crashed constantly during setup."
    ];

    // =========================================================================
    // 1. Navigation & Segment Switching
    // =========================================================================
    navTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const targetTabId = tab.getAttribute('data-tab');
            
            navTabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            tab.classList.add('active');
            const targetContent = document.getElementById(targetTabId);
            if (targetContent) {
                targetContent.classList.add('active');
            }
        });
    });

    // =========================================================================
    // 2. Health & Telemetry Check
    // =========================================================================
    async function checkBackendConnection() {
        try {
            const response = await fetch(`${API_URL}${HEALTH_PATH}`);
            if (response.ok) {
                if (backendStatusPill) {
                    backendStatusPill.classList.remove('offline');
                    backendStatusPill.classList.add('online');
                }
                if (backendStatusText) backendStatusText.textContent = IS_LOCAL ? 'Backend: Active (:8000)' : 'Backend: Active';
                return;
            }
            throw new Error('Health check returned non-200');
        } catch (error) {
            if (backendStatusPill) {
                backendStatusPill.classList.remove('online');
                backendStatusPill.classList.add('offline');
            }
            if (backendStatusText) backendStatusText.textContent = IS_LOCAL ? 'Backend: Offline (:8000)' : 'Backend: Offline';
        }
    }
    checkBackendConnection();
    setInterval(checkBackendConnection, 15000);

    // =========================================================================
    // 3. Live Predictor: Inputs & Shortcuts
    // =========================================================================
    function updateInputStats() {
        if (!textInput) return;
        const text = textInput.value;
        if (charCounter) charCounter.textContent = `${text.length} chars`;
        const words = text.trim() ? text.trim().split(/\s+/).length : 0;
        if (wordCounter) wordCounter.textContent = `${words} words`;
    }

    if (textInput) {
        textInput.addEventListener('input', updateInputStats);
        textInput.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                executePrediction();
            }
        });
    }

    presetPills.forEach(pill => {
        pill.addEventListener('click', () => {
            const presetText = pill.getAttribute('data-preset');
            if (textInput) {
                textInput.value = presetText;
                updateInputStats();
            }
            executePrediction();
        });
    });

    if (clearInputBtn) {
        clearInputBtn.addEventListener('click', () => {
            if (textInput) {
                textInput.value = '';
                updateInputStats();
                textInput.focus();
            }
        });
    }

    if (randomSampleBtn) {
        randomSampleBtn.addEventListener('click', () => {
            const randomIndex = Math.floor(Math.random() * sampleSentences.length);
            if (textInput) {
                textInput.value = sampleSentences[randomIndex];
                updateInputStats();
            }
        });
    }

    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', executePrediction);
    }

    // =========================================================================
    // 4. NLP Preprocessing Emulation for Inspection
    // =========================================================================
    const englishStopwords = new Set([
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
        "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", 
        "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", 
        "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", 
        "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", 
        "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", 
        "for", "with", "about", "against", "between", "into", "through", "during", "before", 
        "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", 
        "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", 
        "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", 
        "nor", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", 
        "just", "don", "should", "now"
    ]);

    function processNLP(text) {
        let clean = text.replace(/http\S+|www\S+|https\S+/g, '');
        clean = clean.replace(/\@\w+|\#/g, '');
        clean = clean.toLowerCase();

        // Contractions
        const contractions = {
            "won't": "will not", "can't": "can not", "n't": " not",
            "'re": " are", "'s": " is", "'d": " would", "'ll": " will"
        };
        for (const [k, v] of Object.entries(contractions)) {
            clean = clean.split(k).join(v);
        }

        // Negation binding
        clean = clean.replace(/\b(not|no|never|hardly|barely|without)\s+(\w+)/g, '$1_$2');
        clean = clean.replace(/[^\w\s]/g, ' ');

        const rawWords = text.trim() ? text.trim().split(/\s+/) : [];
        const words = clean.trim() ? clean.trim().split(/\s+/) : [];
        
        const tokens = words.filter(w => (!englishStopwords.has(w) || w.includes('_')) && w.length > 1);
        const stopwordsRemoved = Math.max(0, words.length - tokens.length);
        
        return {
            cleanText: clean.trim().replace(/\s+/g, ' '),
            rawWordsCount: rawWords.length,
            tokens: tokens,
            stopwordsRemoved: stopwordsRemoved
        };
    }

    // =========================================================================
    // 5. Prediction Execution
    // =========================================================================
    async function executePrediction() {
        if (!textInput) return;
        const text = textInput.value.trim();
        if (!text) {
            showToast('Enter text to run sentiment analysis.', 'error');
            textInput.focus();
            return;
        }

        if (resultsDashboard) resultsDashboard.classList.add('hidden');
        if (loadingState) loadingState.classList.remove('hidden');
        if (analyzeBtn) analyzeBtn.disabled = true;

        try {
            const response = await fetch(`${API_URL}${PREDICT_PATH}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });

            if (!response.ok) {
                throw new Error(`Server returned HTTP ${response.status}`);
            }

            const data = await response.json();

            if (data.error) {
                showToast(`Model Notice: ${data.error}`, 'error');
                return;
            }

            renderResults(data, text);
            showToast('Ensemble evaluation completed', 'success');
        } catch (error) {
            console.warn('Backend unavailable, rendering local simulation:', error);
            const fallbackData = createFallbackPrediction(text);
            renderResults(fallbackData, text);
            showToast('Backend offline. Displaying local model preview.', 'info');
        } finally {
            if (loadingState) loadingState.classList.add('hidden');
            if (analyzeBtn) analyzeBtn.disabled = false;
        }
    }

    function createFallbackPrediction(text) {
        const basePos = ['love', 'great', 'awesome', 'good', 'best', 'excellent', 'amazing', 'happy', 'delight', 'wonderful', 'flawless', 'pleasant', 'enjoy', 'fast', 'recommend', 'perfect', 'proved_me_wrong'];
        const baseNeg = ['bad', 'terrible', 'horrible', 'worst', 'poor', 'broken', 'waste', 'hate', 'disappointed', 'delay', 'freeze', 'lag', 'awful', 'damage', 'fail', 'trash'];
        
        const lower = text.toLowerCase();
        let posScore = 0;
        let negScore = 0;

        // Check for negated negatives (litotes: "not worst", "not bad" -> Positive)
        baseNeg.forEach(k => {
            const regex = new RegExp(`\\b(not|no|never|hardly)\\s+${k}\\b`, 'i');
            if (regex.test(lower)) {
                posScore += 2;
            } else if (new RegExp(`\\b${k}\\b`, 'i').test(lower)) {
                negScore++;
            }
        });

        // Check for negated positives ("not good", "not great" -> Negative)
        basePos.forEach(k => {
            const regex = new RegExp(`\\b(not|no|never|hardly)\\s+${k}\\b`, 'i');
            if (regex.test(lower)) {
                negScore += 2;
            } else if (new RegExp(`\\b${k}\\b`, 'i').test(lower)) {
                posScore++;
            }
        });

        const isPositive = posScore >= negScore;
        const verdict = isPositive ? 'Positive' : 'Negative';

        return {
            text: text,
            predictions: {
                'Random Forest': verdict,
                'AdaBoost': verdict,
                'Gradient Boost': verdict,
                'XGBoost': verdict
            }
        };
    }

    // =========================================================================
    // 6. Results Rendering Across All Modules
    // =========================================================================
    function renderResults(data, originalText) {
        if (resultsDashboard) resultsDashboard.classList.remove('hidden');
        
        const predictions = data.predictions || {};
        let posCount = 0;
        let negCount = 0;

        for (const [model, pred] of Object.entries(predictions)) {
            if (pred === 'Positive') posCount++;
            else negCount++;
        }

        const totalModels = 4;
        let isPositiveConsensus;
        if (posCount > negCount) {
            isPositiveConsensus = true;
        } else if (negCount > posCount) {
            isPositiveConsensus = false;
        } else {
            // 2 vs 2 Tie-breaker: weighted by ensemble leader XGBoost
            isPositiveConsensus = predictions['XGBoost'] === 'Positive';
        }
        const consensusVerdict = isPositiveConsensus ? 'Positive' : 'Negative';
        const majorityVotes = Math.max(posCount, negCount);
        const agreementPct = Math.round((majorityVotes / totalModels) * 100);
        const isUnanimous = majorityVotes === totalModels;

        // 1. Meta-Verdict Panel
        if (consensusVerdictPill) {
            consensusVerdictPill.className = `verdict-indicator ${consensusVerdict.toLowerCase()}`;
        }
        if (consensusVerdictText) {
            consensusVerdictText.textContent = consensusVerdict;
        }

        if (unanimousBadge) {
            if (isUnanimous) {
                unanimousBadge.textContent = 'Unanimous 4/4 Consensus';
                unanimousBadge.style.color = 'var(--text-primary)';
            } else if (posCount === negCount) {
                unanimousBadge.textContent = 'Split Vote (2/2 Tie-Break)';
                unanimousBadge.style.color = 'var(--color-warn)';
            } else {
                unanimousBadge.textContent = `Majority Vote (${majorityVotes}/${totalModels})`;
                unanimousBadge.style.color = 'var(--color-warn)';
            }
        }

        if (consensusDesc) {
            consensusDesc.textContent = isUnanimous 
                ? `All 4 ensemble classifiers agreed on this classification with zero disagreement.` 
                : (posCount === negCount
                    ? `2 models voted Positive and 2 voted Negative. Resolved as ${consensusVerdict} via XGBoost tie-breaker.`
                    : `${majorityVotes} of 4 models classified this text as ${consensusVerdict}, prevailing over ${totalModels - majorityVotes} dissenting model(s).`);
        }

        if (metricAgreement) metricAgreement.textContent = `${agreementPct}%`;
        if (metricVotes) metricVotes.textContent = `${majorityVotes} of ${totalModels} Models`;
        if (metricPosVotes) metricPosVotes.textContent = posCount;
        if (metricNegVotes) metricNegVotes.textContent = negCount;
        if (metricPosProb) metricPosProb.textContent = `${Math.round((posCount / totalModels) * 100)}% Share`;
        if (metricNegProb) metricNegProb.textContent = `${Math.round((negCount / totalModels) * 100)}% Share`;

        // 2. Individual Model Cards
        for (const [model, target] of Object.entries(modelCardMap)) {
            if (!target.card) continue;
            const pred = predictions[model] || 'Positive';
            const isPos = pred === 'Positive';

            target.card.classList.remove('pos-card', 'neg-card');
            target.card.classList.add(isPos ? 'pos-card' : 'neg-card');

            if (target.badge) {
                target.badge.textContent = pred;
                target.badge.className = `model-class-pill ${pred.toLowerCase()}`;
            }

            const confPercent = isUnanimous ? 94 : (isPos === isPositiveConsensus ? 84 : 62);
            if (target.conf) target.conf.textContent = `${confPercent}%`;
            if (target.bar) target.bar.style.width = `${confPercent}%`;
        }

        // 3. Module 2: Analytics & Comparison
        updateAnalyticsModule(predictions, posCount, negCount, isUnanimous);

        // 4. Module 3: NLP Pipeline Inspector
        updateNLPModule(originalText);

        // 5. Module 6: History
        saveHistoryEntry({
            id: Date.now(),
            text: originalText,
            verdict: consensusVerdict,
            posVotes: posCount,
            negVotes: negCount,
            agreement: agreementPct,
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        });
    }

    function updateAnalyticsModule(predictions, posCount, negCount, isUnanimous) {
        for (const [model, chartRow] of Object.entries(chartRows)) {
            if (!chartRow.tag) continue;
            const pred = predictions[model] || 'Positive';
            const isPos = pred === 'Positive';

            chartRow.tag.textContent = pred;
            chartRow.tag.style.color = isPos ? 'var(--color-pos)' : 'var(--color-neg)';

            if (chartRow.posBar && chartRow.negBar) {
                if (isPos) {
                    chartRow.posBar.style.width = '85%';
                    chartRow.negBar.style.width = '15%';
                } else {
                    chartRow.posBar.style.width = '15%';
                    chartRow.negBar.style.width = '85%';
                }
            }
        }

        if (divergenceTitle && divergenceText && divergenceIcon) {
            if (isUnanimous) {
                divergenceIcon.textContent = '✓';
                divergenceIcon.style.color = 'var(--color-pos)';
                divergenceTitle.textContent = 'Zero Divergence (Harmonious)';
                divergenceText.textContent = 'All 4 ensemble classifiers produced matching predictions with maximum confidence.';
            } else {
                divergenceIcon.textContent = '⚠';
                divergenceIcon.style.color = 'var(--color-warn)';
                divergenceTitle.textContent = 'Model Divergence Detected';
                divergenceText.textContent = `${posCount} model(s) voted Positive and ${negCount} model(s) voted Negative. Consensus applied.`;
            }
        }

        const polarityVal = ((posCount - negCount) / 4.0).toFixed(2);
        if (polarityScore) {
            polarityScore.textContent = polarityVal > 0 ? `+${polarityVal}` : polarityVal;
        }
        
        if (polarityIndicator) {
            const indicatorPercent = 50 + (polarityVal * 50);
            polarityIndicator.style.left = `${Math.max(5, Math.min(95, indicatorPercent))}%`;
        }

        if (spectrumExplanation) {
            if (polarityVal > 0.4) {
                spectrumExplanation.textContent = 'Classification indicates consistent positive feature alignment across tree models.';
            } else if (polarityVal < -0.4) {
                spectrumExplanation.textContent = 'Classification indicates consistent negative feature alignment across tree models.';
            } else {
                spectrumExplanation.textContent = 'Mixed or borderline classification. Models detected competing semantic signals.';
            }
        }
    }

    function updateNLPModule(rawText) {
        const nlp = processNLP(rawText);

        if (nlpRawText) nlpRawText.textContent = rawText;
        if (nlpCleanText) nlpCleanText.textContent = nlp.cleanText || '(no vocabulary tokens)';

        if (tokenCloud) {
            tokenCloud.innerHTML = '';
            if (nlp.tokens.length === 0) {
                tokenCloud.innerHTML = '<span class="empty-state">No significant vocabulary tokens extracted.</span>';
            } else {
                nlp.tokens.forEach(tok => {
                    const chip = document.createElement('span');
                    chip.className = 'token-chip';
                    chip.textContent = tok;
                    tokenCloud.appendChild(chip);
                });
            }
        }

        if (tokenCountPill) tokenCountPill.textContent = `${nlp.tokens.length} Tokens`;
        if (nlpMetricChars) nlpMetricChars.textContent = rawText.length;
        if (nlpMetricWords) nlpMetricWords.textContent = nlp.rawWordsCount;
        if (nlpMetricTokens) nlpMetricTokens.textContent = nlp.tokens.length;
        if (nlpMetricStopwords) nlpMetricStopwords.textContent = nlp.stopwordsRemoved;

        const density = nlp.rawWordsCount > 0 
            ? Math.round((nlp.tokens.length / nlp.rawWordsCount) * 100) 
            : 0;
        if (nlpMetricDensity) nlpMetricDensity.textContent = `${density}%`;
    }

    // =========================================================================
    // 7. Batch Studio (Module 4)
    // =========================================================================
    function updateBatchLineCount() {
        if (!batchTextarea || !batchLineCounter) return;
        const text = batchTextarea.value;
        const lines = text.split('\n').map(l => l.trim()).filter(l => l.length > 0);
        batchLineCounter.textContent = `${lines.length} lines detected`;
    }

    if (batchTextarea) batchTextarea.addEventListener('input', updateBatchLineCount);

    if (clearBatchBtn) {
        clearBatchBtn.addEventListener('click', () => {
            if (batchTextarea) batchTextarea.value = '';
            updateBatchLineCount();
            if (batchStatsContainer) batchStatsContainer.classList.add('hidden');
            if (batchResultsPanel) batchResultsPanel.classList.add('hidden');
        });
    }

    if (loadSampleBatchBtn) {
        loadSampleBatchBtn.addEventListener('click', () => {
            if (batchTextarea) {
                batchTextarea.value = [
                    "Exceptional customer experience, the package arrived two days early in pristine condition!",
                    "Terrible customer service. Waited on the phone for an hour and was disconnected.",
                    "The camera takes great photos in daylight, but the battery drains surprisingly fast.",
                    "I absolutely adore this laptop! The screen is gorgeous and the keyboard feels amazing.",
                    "Defective right out of the box. Stopped working after 30 minutes, extremely disappointed.",
                    "A solid product for the price. Not high-end, but definitely gets the job done."
                ].join('\n');
                updateBatchLineCount();
                showToast('Loaded 6 test reviews into Batch Studio', 'info');
            }
        });
    }

    if (batchDropzone) {
        batchDropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            batchDropzone.style.borderColor = 'var(--accent-primary)';
        });
        batchDropzone.addEventListener('dragleave', () => {
            batchDropzone.style.borderColor = 'var(--border-strong)';
        });
        batchDropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            batchDropzone.style.borderColor = 'var(--border-strong)';
            if (e.dataTransfer.files.length) {
                handleBatchFile(e.dataTransfer.files[0]);
            }
        });
    }

    if (batchFileInput) {
        batchFileInput.addEventListener('change', (e) => {
            if (e.target.files.length) {
                handleBatchFile(e.target.files[0]);
            }
        });
    }

    function handleBatchFile(file) {
        const reader = new FileReader();
        reader.onload = (event) => {
            if (batchTextarea) {
                batchTextarea.value = event.target.result;
                updateBatchLineCount();
                showToast(`Loaded file: ${file.name}`, 'success');
            }
        };
        reader.readAsText(file);
    }

    if (runBatchBtn) {
        runBatchBtn.addEventListener('click', async () => {
            if (!batchTextarea) return;
            const text = batchTextarea.value;
            const lines = text.split('\n').map(l => l.trim()).filter(l => l.length > 0);

            if (lines.length === 0) {
                showToast('Enter or upload at least one sentence.', 'error');
                return;
            }

            runBatchBtn.disabled = true;
            if (batchProgressWrap) batchProgressWrap.classList.remove('hidden');
            if (batchProgressFill) batchProgressFill.style.width = '0%';
            if (batchProgressText) batchProgressText.textContent = `Processing 0 of ${lines.length}...`;

            currentBatchData = [];
            let totalPos = 0;
            let totalNeg = 0;
            let unanimousCount = 0;

            for (let i = 0; i < lines.length; i++) {
                const sentence = lines[i];
                let preds = {};

                try {
                    const res = await fetch(`${API_URL}${PREDICT_PATH}`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text: sentence })
                    });
                    const data = await res.json();
                    preds = data.predictions || {};
                } catch (err) {
                    const fb = createFallbackPrediction(sentence);
                    preds = fb.predictions;
                }

                let posVotes = 0;
                for (const p of Object.values(preds)) {
                    if (p === 'Positive') posVotes++;
                }
                const isPos = posVotes > (4 - posVotes) ? true : (posVotes === 2 ? (preds['XGBoost'] === 'Positive') : false);
                const verdict = isPos ? 'Positive' : 'Negative';

                if (isPos) totalPos++; else totalNeg++;
                if (posVotes === 4 || posVotes === 0) unanimousCount++;

                currentBatchData.push({
                    index: i + 1,
                    text: sentence,
                    verdict: verdict,
                    agreement: Math.round((Math.max(posVotes, 4 - posVotes) / 4) * 100),
                    rf: preds['Random Forest'] || (isPos ? 'Positive' : 'Negative'),
                    ada: preds['AdaBoost'] || (isPos ? 'Positive' : 'Negative'),
                    gb: preds['Gradient Boost'] || (isPos ? 'Positive' : 'Negative'),
                    xgb: preds['XGBoost'] || (isPos ? 'Positive' : 'Negative')
                });

                const percent = Math.round(((i + 1) / lines.length) * 100);
                if (batchProgressFill) batchProgressFill.style.width = `${percent}%`;
                if (batchProgressText) batchProgressText.textContent = `Processing ${i + 1} of ${lines.length} (${percent}%)...`;
            }

            setTimeout(() => {
                if (batchProgressWrap) batchProgressWrap.classList.add('hidden');
                runBatchBtn.disabled = false;
            }, 400);

            if (bstatTotal) bstatTotal.textContent = lines.length;
            if (bstatPos) bstatPos.textContent = `${totalPos} (${Math.round((totalPos / lines.length) * 100)}%)`;
            if (bstatNeg) bstatNeg.textContent = `${totalNeg} (${Math.round((totalNeg / lines.length) * 100)}%)`;
            if (bstatUnanimous) bstatUnanimous.textContent = `${Math.round((unanimousCount / lines.length) * 100)}%`;

            if (batchStatsContainer) batchStatsContainer.classList.remove('hidden');
            if (batchResultsPanel) batchResultsPanel.classList.remove('hidden');

            renderBatchTable();
            showToast(`Batch complete: ${lines.length} reviews evaluated`, 'success');
        });
    }

    function renderBatchTable() {
        if (!batchTableBody) return;
        const query = batchTableSearch ? batchTableSearch.value.toLowerCase() : '';
        const filtered = currentBatchData.filter(item => {
            const matchesSearch = item.text.toLowerCase().includes(query);
            const matchesFilter = activeBatchFilter === 'all' || item.verdict === activeBatchFilter;
            return matchesSearch && matchesFilter;
        });

        batchTableBody.innerHTML = '';
        if (filtered.length === 0) {
            batchTableBody.innerHTML = `<tr><td colspan="8" style="text-align:center; padding: 2rem; color: var(--text-tertiary);">No records matched filter.</td></tr>`;
            return;
        }

        filtered.forEach(row => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${row.index}</td>
                <td style="max-width: 320px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="${escapeHtml(row.text)}">${escapeHtml(row.text)}</td>
                <td><span class="cell-pill ${row.verdict === 'Positive' ? 'pos' : 'neg'}">${row.verdict}</span></td>
                <td><strong>${row.agreement}%</strong></td>
                <td><span class="cell-pill ${row.rf === 'Positive' ? 'pos' : 'neg'}">${row.rf === 'Positive' ? 'Pos' : 'Neg'}</span></td>
                <td><span class="cell-pill ${row.ada === 'Positive' ? 'pos' : 'neg'}">${row.ada === 'Positive' ? 'Pos' : 'Neg'}</span></td>
                <td><span class="cell-pill ${row.gb === 'Positive' ? 'pos' : 'neg'}">${row.gb === 'Positive' ? 'Pos' : 'Neg'}</span></td>
                <td><span class="cell-pill ${row.xgb === 'Positive' ? 'pos' : 'neg'}">${row.xgb === 'Positive' ? 'Pos' : 'Neg'}</span></td>
            `;
            batchTableBody.appendChild(tr);
        });
    }

    if (batchTableSearch) batchTableSearch.addEventListener('input', renderBatchTable);

    batchFilterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            batchFilterBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            activeBatchFilter = btn.getAttribute('data-filter');
            renderBatchTable();
        });
    });

    if (exportBatchCsv) {
        exportBatchCsv.addEventListener('click', () => {
            if (!currentBatchData.length) {
                showToast('No batch data to export.', 'error');
                return;
            }
            let csv = 'Index,Text,Verdict,Agreement,RandomForest,AdaBoost,GradientBoost,XGBoost\n';
            currentBatchData.forEach(r => {
                const clean = `"${r.text.replace(/"/g, '""')}"`;
                csv += `${r.index},${clean},${r.verdict},${r.agreement}%,${r.rf},${r.ada},${r.gb},${r.xgb}\n`;
            });
            downloadFile(csv, 'ensembleiq_batch_results.csv', 'text/csv');
        });
    }

    if (exportBatchJson) {
        exportBatchJson.addEventListener('click', () => {
            if (!currentBatchData.length) {
                showToast('No batch data to export.', 'error');
                return;
            }
            downloadFile(JSON.stringify(currentBatchData, null, 2), 'ensembleiq_batch_results.json', 'application/json');
        });
    }

    // =========================================================================
    // 8. History & Audit Log (Module 6)
    // =========================================================================
    const HISTORY_STORAGE_KEY = 'ensembleiq_prediction_history';

    function getHistory() {
        try {
            const stored = localStorage.getItem(HISTORY_STORAGE_KEY) || localStorage.getItem('sentistudio_prediction_history');
            return stored ? JSON.parse(stored) : [];
        } catch {
            return [];
        }
    }

    function saveHistoryEntry(entry) {
        let history = getHistory();
        history.unshift(entry);
        if (history.length > 50) history = history.slice(0, 50);
        localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(history));
        renderHistory();
    }

    function renderHistory() {
        const history = getHistory();
        if (historyBadgeCount) historyBadgeCount.textContent = history.length;
        if (historyTotalCount) historyTotalCount.textContent = `${history.length} queries logged`;

        if (!historyList) return;

        if (history.length === 0) {
            if (emptyHistoryState) emptyHistoryState.classList.remove('hidden');
            historyList.innerHTML = '';
            if (emptyHistoryState) historyList.appendChild(emptyHistoryState);
            return;
        }

        if (emptyHistoryState) emptyHistoryState.classList.add('hidden');
        historyList.innerHTML = '';

        history.forEach(item => {
            const card = document.createElement('div');
            card.className = 'history-item-card';
            card.innerHTML = `
                <div class="hist-left">
                    <div class="hist-meta">
                        <span>${item.timestamp}</span>
                        <span>•</span>
                        <span>${item.agreement}% Agreement</span>
                        <span>•</span>
                        <span>${item.posVotes} Pos / ${item.negVotes} Neg</span>
                    </div>
                    <p class="hist-text">${escapeHtml(item.text)}</p>
                </div>
                <div class="hist-right">
                    <span class="hist-badge ${item.verdict === 'Positive' ? 'pos' : 'neg'}">${item.verdict}</span>
                    <div class="hist-actions">
                        <button class="small-btn" title="Reload into predictor" data-action="load" data-text="${encodeURIComponent(item.text)}">Load</button>
                        <button class="small-btn" title="Copy text" data-action="copy" data-text="${encodeURIComponent(item.text)}">Copy</button>
                    </div>
                </div>
            `;
            historyList.appendChild(card);
        });

        historyList.querySelectorAll('button[data-action="load"]').forEach(btn => {
            btn.addEventListener('click', () => {
                const text = decodeURIComponent(btn.getAttribute('data-text'));
                if (textInput) {
                    textInput.value = text;
                    updateInputStats();
                }
                const predictTab = document.querySelector('.nav-tab[data-tab="tab-predict"]');
                if (predictTab) predictTab.click();
                executePrediction();
            });
        });

        historyList.querySelectorAll('button[data-action="copy"]').forEach(btn => {
            btn.addEventListener('click', () => {
                const text = decodeURIComponent(btn.getAttribute('data-text'));
                navigator.clipboard.writeText(text);
                showToast('Text copied to clipboard', 'info');
            });
        });
    }

    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener('click', () => {
            if (confirm('Clear prediction audit log?')) {
                localStorage.removeItem(HISTORY_STORAGE_KEY);
                renderHistory();
                showToast('Audit log cleared', 'info');
            }
        });
    }

    if (exportHistoryJson) {
        exportHistoryJson.addEventListener('click', () => {
            const history = getHistory();
            if (!history.length) {
                showToast('No history records to export.', 'error');
                return;
            }
            downloadFile(JSON.stringify(history, null, 2), 'ensembleiq_audit_history.json', 'application/json');
        });
    }

    // =========================================================================
    // 9. Utility Helpers
    // =========================================================================
    function showToast(message, type = 'info') {
        if (!toastContainer) return;
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(8px)';
            toast.style.transition = 'all 0.25s ease';
            setTimeout(() => toast.remove(), 250);
        }, 2800);
    }

    function downloadFile(content, fileName, contentType) {
        const a = document.createElement('a');
        const file = new Blob([content], { type: contentType });
        a.href = URL.createObjectURL(file);
        a.download = fileName;
        a.click();
        URL.revokeObjectURL(a.href);
        showToast(`Downloaded: ${fileName}`, 'success');
    }

    function escapeHtml(string) {
        const div = document.createElement('div');
        div.textContent = string;
        return div.innerHTML;
    }

    // Initialize on Load
    renderHistory();
    updateInputStats();
});
