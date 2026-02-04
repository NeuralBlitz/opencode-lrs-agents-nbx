import * as vscode from 'vscode';
import axios, { AxiosResponse } from 'axios';
import * as WebSocket from 'ws';

interface LRSConfig {
    serverUrl: string;
    enableTelemetry: boolean;
    autoAnalyze: boolean;
    precisionLevel: 'low' | 'medium' | 'high' | 'adaptive';
    maxFileSize: number;
    supportedLanguages: string[];
}

interface LRSAnalysisResult {
    success: boolean;
    data?: any;
    error?: string;
    execution_time?: number;
    precision_used?: number;
    recommendations?: string[];
}

interface LRSPlanResult {
    success: boolean;
    goals?: string[];
    tasks?: Array<{
        id: string;
        description: string;
        complexity: number;
        dependencies: string[];
        estimated_effort: string;
    }>;
    execution_steps?: string[];
    risk_assessment?: any;
    free_energy?: number;
}

interface LRSEvaluationResult {
    success: boolean;
    rankings?: Array<{
        strategy: string;
        score: number;
        confidence: number;
        reasoning: string;
    }>;
    recommended_strategy?: string;
}

export function activate(context: vscode.ExtensionContext) {
    console.log('OpenCode LRS Integration extension is now active! 🚀');

    // Initialize LRS client
    const lrsClient = new LRSClient();

    // Register commands
    context.subscriptions.push(
        vscode.commands.registerCommand('opencode.lrs.analyze', async () => {
            await analyzeCommand(lrsClient);
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('opencode.lrs.refactor', async () => {
            await refactorCommand(lrsClient);
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('opencode.lrs.plan', async () => {
            await planCommand(lrsClient);
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('opencode.lrs.evaluate', async () => {
            await evaluateCommand(lrsClient);
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('opencode.lrs.stats', async () => {
            await statsCommand(lrsClient);
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('opencode.lrs.benchmark', async () => {
            await benchmarkCommand(lrsClient);
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('opencode.lrs.configure', async () => {
            await configureCommand();
        })
    );

    // Register auto-analyze on save if enabled
    const config = getLRSConfig();
    if (config.autoAnalyze) {
        context.subscriptions.push(
            vscode.workspace.onDidSaveTextDocument(async (document) => {
                if (isLanguageSupported(document.languageId, config)) {
                    await autoAnalyzeDocument(document, lrsClient);
                }
            })
        );
    }

    // Register code actions provider
    context.subscriptions.push(
        vscode.languages.registerCodeActionsProvider(
            getSupportedLanguageSelector(),
            new LRSCodelensProvider(lrsClient),
            {
                providedCodeActionKinds: LRSCodelensProvider.providedCodeActionKinds
            }
        )
    );

    // Status bar item
    const statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    statusBarItem.command = 'opencode.lrs.stats';
    updateStatusBar(statusBarItem, lrsClient);
    context.subscriptions.push(statusBarItem);

    // Update status bar periodically
    const statusUpdateInterval = setInterval(() => {
        updateStatusBar(statusBarItem, lrsClient);
    }, 30000); // Update every 30 seconds

    context.subscriptions.push({
        dispose: () => clearInterval(statusUpdateInterval)
    });
}

export function deactivate() {
    console.log('OpenCode LRS Integration extension deactivated');
}

class LRSClient {
    private config: LRSConfig;
    private websocket: WebSocket | null = null;

    constructor() {
        this.config = getLRSConfig();
    }

    async analyzeCode(code: string, language: string, context?: string): Promise<LRSAnalysisResult> {
        try {
            const response: AxiosResponse = await axios.post(`${this.config.serverUrl}/api/lrs/analyze`, {
                code,
                language,
                context,
                precision_level: this.config.precisionLevel
            }, {
                timeout: 30000
            });

            return {
                success: true,
                data: response.data,
                execution_time: response.data.execution_time,
                precision_used: response.data.precision_used,
                recommendations: response.data.recommendations
            };
        } catch (error: any) {
            return {
                success: false,
                error: error.message || 'Analysis failed'
            };
        }
    }

    async refactorCode(code: string, language: string, instructions?: string): Promise<LRSAnalysisResult> {
        try {
            const response: AxiosResponse = await axios.post(`${this.config.serverUrl}/api/lrs/refactor`, {
                code,
                language,
                instructions,
                precision_level: this.config.precisionLevel
            }, {
                timeout: 30000
            });

            return {
                success: true,
                data: response.data,
                execution_time: response.data.execution_time,
                precision_used: response.data.precision_used,
                recommendations: response.data.recommendations
            };
        } catch (error: any) {
            return {
                success: false,
                error: error.message || 'Refactoring failed'
            };
        }
    }

    async generatePlan(description: string, context?: any): Promise<LRSPlanResult> {
        try {
            const response: AxiosResponse = await axios.post(`${this.config.serverUrl}/api/lrs/plan`, {
                description,
                context,
                precision_level: this.config.precisionLevel
            }, {
                timeout: 45000
            });

            return {
                success: true,
                goals: response.data.goals,
                tasks: response.data.tasks,
                execution_steps: response.data.execution_steps,
                risk_assessment: response.data.risk_assessment,
                free_energy: response.data.free_energy
            };
        } catch (error: any) {
            return {
                success: false
            };
        }
    }

    async evaluateStrategies(task: string, strategies: string[]): Promise<LRSEvaluationResult> {
        try {
            const response: AxiosResponse = await axios.post(`${this.config.serverUrl}/api/lrs/evaluate`, {
                task,
                strategies,
                precision_level: this.config.precisionLevel
            }, {
                timeout: 30000
            });

            return {
                success: true,
                rankings: response.data.rankings,
                recommended_strategy: response.data.recommended_strategy
            };
        } catch (error: any) {
            return {
                success: false
            };
        }
    }

    async getStats(): Promise<any> {
        try {
            const response: AxiosResponse = await axios.get(`${this.config.serverUrl}/api/lrs/stats`);
            return response.data;
        } catch (error: any) {
            return { error: error.message || 'Failed to get stats' };
        }
    }

    async runBenchmarks(): Promise<any> {
        try {
            const response: AxiosResponse = await axios.post(`${this.config.serverUrl}/api/benchmarks/run`, {}, {
                timeout: 120000 // 2 minutes for benchmarks
            });
            return response.data;
        } catch (error: any) {
            return { error: error.message || 'Benchmark execution failed' };
        }
    }

    connectWebSocket(): void {
        if (this.websocket) {
            return;
        }

        try {
            const wsUrl = this.config.serverUrl.replace('http', 'ws') + '/ws';
            this.websocket = new WebSocket(wsUrl);

            this.websocket.onopen = () => {
                console.log('LRS WebSocket connected');
            };

            this.websocket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data.toString());
                    this.handleWebSocketMessage(data);
                } catch (error) {
                    console.error('Failed to parse WebSocket message:', error);
                }
            };

            this.websocket.onclose = () => {
                console.log('LRS WebSocket disconnected');
                this.websocket = null;
                // Auto-reconnect after 5 seconds
                setTimeout(() => this.connectWebSocket(), 5000);
            };

            this.websocket.onerror = (error) => {
                console.error('LRS WebSocket error:', error);
            };
        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
        }
    }

    private handleWebSocketMessage(data: any): void {
        // Handle real-time updates from LRS server
        if (data.type === 'precision_update') {
            console.log('Precision updated:', data.precision);
        } else if (data.type === 'analysis_complete') {
            vscode.window.showInformationMessage(`Analysis complete: ${data.result}`);
        } else if (data.type === 'error') {
            vscode.window.showErrorMessage(`LRS Error: ${data.message}`);
        }
    }

    disconnectWebSocket(): void {
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
    }
}

function getLRSConfig(): LRSConfig {
    const config = vscode.workspace.getConfiguration('opencode.lrs');
    return {
        serverUrl: config.get('serverUrl', 'http://localhost:8000'),
        enableTelemetry: config.get('enableTelemetry', true),
        autoAnalyze: config.get('autoAnalyze', false),
        precisionLevel: config.get('precisionLevel', 'adaptive'),
        maxFileSize: config.get('maxFileSize', 1048576),
        supportedLanguages: config.get('supportedLanguages', [
            'javascript', 'typescript', 'python', 'java', 'cpp', 'csharp', 'go', 'rust'
        ])
    };
}

function isLanguageSupported(languageId: string, config: LRSConfig): boolean {
    return config.supportedLanguages.includes(languageId);
}

function getSupportedLanguageSelector(): vscode.DocumentSelector {
    const config = getLRSConfig();
    return config.supportedLanguages.map(lang => ({ language: lang }));
}

async function analyzeCommand(lrsClient: LRSClient): Promise<void> {
    const activeEditor = vscode.window.activeTextEditor;
    if (!activeEditor) {
        vscode.window.showErrorMessage('No active editor found');
        return;
    }

    const document = activeEditor.document;
    const config = getLRSConfig();

    if (!isLanguageSupported(document.languageId, config)) {
        vscode.window.showWarningMessage(`Language '${document.languageId}' is not supported by LRS`);
        return;
    }

    await vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: "Analyzing with LRS...",
        cancellable: true
    }, async (progress, token) => {
        progress.report({ increment: 10, message: "Sending code to LRS server..." });

        const code = document.getText();
        const result = await lrsClient.analyzeCode(code, document.languageId);

        progress.report({ increment: 90, message: "Processing analysis results..." });

        if (result.success && result.data) {
            // Show analysis results
            const analysisData = result.data;
            const executionTime = result.execution_time || 0;
            const precision = result.precision_used || 0;

            // Create output channel for detailed results
            const outputChannel = vscode.window.createOutputChannel('LRS Analysis');
            outputChannel.clear();
            outputChannel.appendLine('🤖 OpenCode LRS Analysis Results');
            outputChannel.appendLine('=' * 40);
            outputChannel.appendLine(`📊 Execution Time: ${executionTime.toFixed(3)}s`);
            outputChannel.appendLine(`🎯 Precision Used: ${precision.toFixed(3)}`);
            outputChannel.appendLine(`📝 Analysis: ${JSON.stringify(analysisData, null, 2)}`);
            outputChannel.show();

            // Show summary notification
            const recommendations = result.recommendations || [];
            const recCount = recommendations.length;

            vscode.window.showInformationMessage(
                `LRS Analysis Complete (${executionTime.toFixed(2)}s) - ${recCount} recommendations found`
            );

            // Show recommendations if any
            if (recCount > 0) {
                const showRecs = await vscode.window.showQuickPick([
                    'Show Recommendations',
                    'Dismiss'
                ], { placeHolder: `${recCount} recommendations available` });

                if (showRecs === 'Show Recommendations') {
                    const recChannel = vscode.window.createOutputChannel('LRS Recommendations');
                    recChannel.clear();
                    recChannel.appendLine('💡 LRS Analysis Recommendations');
                    recChannel.appendLine('=' * 35);
                    recommendations.forEach((rec: string, index: number) => {
                        recChannel.appendLine(`${index + 1}. ${rec}`);
                    });
                    recChannel.show();
                }
            }
        } else {
            vscode.window.showErrorMessage(`LRS Analysis Failed: ${result.error || 'Unknown error'}`);
        }
    });
}

async function refactorCommand(lrsClient: LRSClient): Promise<void> {
    const activeEditor = vscode.window.activeTextEditor;
    if (!activeEditor) {
        vscode.window.showErrorMessage('No active editor found');
        return;
    }

    const selection = activeEditor.selection;
    if (selection.isEmpty) {
        vscode.window.showErrorMessage('Please select code to refactor');
        return;
    }

    const document = activeEditor.document;
    const config = getLRSConfig();

    if (!isLanguageSupported(document.languageId, config)) {
        vscode.window.showWarningMessage(`Language '${document.languageId}' is not supported by LRS`);
        return;
    }

    // Get refactoring instructions from user
    const instructions = await vscode.window.showInputBox({
        prompt: 'Enter refactoring instructions',
        placeHolder: 'e.g., "optimize performance", "improve readability", "extract method"'
    });

    if (!instructions) {
        return;
    }

    await vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: "Refactoring with LRS...",
        cancellable: true
    }, async (progress, token) => {
        progress.report({ increment: 20, message: "Sending code to LRS server..." });

        const selectedCode = document.getText(selection);
        const result = await lrsClient.refactorCode(selectedCode, document.languageId, instructions);

        progress.report({ increment: 80, message: "Applying refactoring suggestions..." });

        if (result.success && result.data) {
            // Apply the refactoring
            const refactoredCode = result.data.refactored_code || result.data;
            const executionTime = result.execution_time || 0;

            // Replace the selected code
            await activeEditor.edit(editBuilder => {
                editBuilder.replace(selection, refactoredCode);
            });

            vscode.window.showInformationMessage(
                `LRS Refactoring Applied (${executionTime.toFixed(2)}s)`
            );

            // Show detailed results if available
            if (result.recommendations && result.recommendations.length > 0) {
                const outputChannel = vscode.window.createOutputChannel('LRS Refactoring Details');
                outputChannel.clear();
                outputChannel.appendLine('🔧 LRS Refactoring Details');
                outputChannel.appendLine('=' * 30);
                result.recommendations.forEach((rec: string, index: number) => {
                    outputChannel.appendLine(`${index + 1}. ${rec}`);
                });
                outputChannel.show();
            }
        } else {
            vscode.window.showErrorMessage(`LRS Refactoring Failed: ${result.error || 'Unknown error'}`);
        }
    });
}

async function planCommand(lrsClient: LRSClient): Promise<void> {
    const description = await vscode.window.showInputBox({
        prompt: 'Describe the development task or feature',
        placeHolder: 'e.g., "Build a user authentication system", "Implement data visualization dashboard"'
    });

    if (!description) {
        return;
    }

    await vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: "Generating development plan with LRS...",
        cancellable: true
    }, async (progress, token) => {
        progress.report({ increment: 30, message: "Analyzing requirements..." });

        const result = await lrsClient.generatePlan(description);

        progress.report({ increment: 70, message: "Creating execution plan..." });

        if (result.success) {
            // Create output channel for the plan
            const outputChannel = vscode.window.createOutputChannel('LRS Development Plan');
            outputChannel.clear();
            outputChannel.appendLine('📋 LRS Development Plan');
            outputChannel.appendLine('=' * 25);
            outputChannel.appendLine(`🎯 Description: ${description}`);
            outputChannel.appendLine('');

            if (result.goals) {
                outputChannel.appendLine('🎯 Goals:');
                result.goals.forEach((goal: string, index: number) => {
                    outputChannel.appendLine(`   ${index + 1}. ${goal}`);
                });
                outputChannel.appendLine('');
            }

            if (result.tasks) {
                outputChannel.appendLine('📝 Tasks:');
                result.tasks.forEach((task: any) => {
                    outputChannel.appendLine(`   • ${task.description}`);
                    outputChannel.appendLine(`     Complexity: ${task.complexity}/10 | Effort: ${task.estimated_effort}`);
                    if (task.dependencies && task.dependencies.length > 0) {
                        outputChannel.appendLine(`     Dependencies: ${task.dependencies.join(', ')}`);
                    }
                    outputChannel.appendLine('');
                });
            }

            if (result.execution_steps) {
                outputChannel.appendLine('🚀 Execution Steps:');
                result.execution_steps.forEach((step: string, index: number) => {
                    outputChannel.appendLine(`   ${index + 1}. ${step}`);
                });
                outputChannel.appendLine('');
            }

            if (result.free_energy !== undefined) {
                outputChannel.appendLine(`⚡ Free Energy Score: ${result.free_energy.toFixed(4)}`);
            }

            outputChannel.show();

            vscode.window.showInformationMessage(
                `LRS Development Plan Generated - ${result.tasks?.length || 0} tasks identified`
            );
        } else {
            vscode.window.showErrorMessage('LRS Planning Failed: Unable to generate development plan');
        }
    });
}

async function evaluateCommand(lrsClient: LRSClient): Promise<void> {
    const task = await vscode.window.showInputBox({
        prompt: 'Describe the task or decision to evaluate',
        placeHolder: 'e.g., "Choose a database for the application"'
    });

    if (!task) {
        return;
    }

    // Get strategies to evaluate
    const strategiesInput = await vscode.window.showInputBox({
        prompt: 'Enter strategies to evaluate (comma-separated)',
        placeHolder: 'e.g., "PostgreSQL, MongoDB, MySQL, SQLite"'
    });

    if (!strategiesInput) {
        return;
    }

    const strategies = strategiesInput.split(',').map(s => s.trim()).filter(s => s.length > 0);

    if (strategies.length < 2) {
        vscode.window.showErrorMessage('Please provide at least 2 strategies to evaluate');
        return;
    }

    await vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: "Evaluating strategies with LRS...",
        cancellable: true
    }, async (progress, token) => {
        progress.report({ increment: 40, message: "Running strategy evaluation..." });

        const result = await lrsClient.evaluateStrategies(task, strategies);

        progress.report({ increment: 60, message: "Processing evaluation results..." });

        if (result.success && result.rankings) {
            // Create output channel for results
            const outputChannel = vscode.window.createOutputChannel('LRS Strategy Evaluation');
            outputChannel.clear();
            outputChannel.appendLine('⚖️ LRS Strategy Evaluation Results');
            outputChannel.appendLine('=' * 35);
            outputChannel.appendLine(`🎯 Task: ${task}`);
            outputChannel.appendLine('');

            outputChannel.appendLine('📊 Rankings:');
            result.rankings.forEach((ranking: any, index: number) => {
                const medal = index === 0 ? '🥇' : index === 1 ? '🥈' : index === 2 ? '🥉' : '📍';
                outputChannel.appendLine(`${medal} ${ranking.strategy}`);
                outputChannel.appendLine(`   Score: ${ranking.score.toFixed(3)} | Confidence: ${(ranking.confidence * 100).toFixed(1)}%`);
                outputChannel.appendLine(`   Reasoning: ${ranking.reasoning}`);
                outputChannel.appendLine('');
            });

            if (result.recommended_strategy) {
                outputChannel.appendLine(`💡 Recommended Strategy: ${result.recommended_strategy}`);
            }

            outputChannel.show();

            const topStrategy = result.rankings[0];
            vscode.window.showInformationMessage(
                `LRS Evaluation Complete - Top Strategy: ${topStrategy.strategy} (${topStrategy.score.toFixed(3)})`
            );
        } else {
            vscode.window.showErrorMessage('LRS Strategy Evaluation Failed');
        }
    });
}

async function statsCommand(lrsClient: LRSClient): Promise<void> {
    await vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: "Fetching LRS statistics...",
        cancellable: false
    }, async (progress, token) => {
        progress.report({ increment: 50, message: "Contacting LRS server..." });

        const stats = await lrsClient.getStats();

        if (stats.error) {
            vscode.window.showErrorMessage(`Failed to get LRS stats: ${stats.error}`);
            return;
        }

        // Create output channel for stats
        const outputChannel = vscode.window.createOutputChannel('LRS System Statistics');
        outputChannel.clear();
        outputChannel.appendLine('📊 LRS System Statistics');
        outputChannel.appendLine('=' * 25);

        // System info
        if (stats.system) {
            outputChannel.appendLine('🖥️ System Information:');
            outputChannel.appendLine(`   Uptime: ${stats.system.uptime || 'N/A'}`);
            outputChannel.appendLine(`   Version: ${stats.system.version || 'N/A'}`);
            outputChannel.appendLine(`   Active Sessions: ${stats.system.active_sessions || 0}`);
            outputChannel.appendLine('');
        }

        // Performance metrics
        if (stats.performance) {
            outputChannel.appendLine('⚡ Performance Metrics:');
            outputChannel.appendLine(`   Avg Response Time: ${stats.performance.avg_response_time?.toFixed(3) || 'N/A'}s`);
            outputChannel.appendLine(`   Total Requests: ${stats.performance.total_requests || 0}`);
            outputChannel.appendLine(`   Cache Hit Rate: ${(stats.performance.cache_hit_rate * 100)?.toFixed(1) || 'N/A'}%`);
            outputChannel.appendLine('');
        }

        // LRS-specific metrics
        if (stats.lrs) {
            outputChannel.appendLine('🧠 LRS Metrics:');
            outputChannel.appendLine(`   Active Agents: ${stats.lrs.active_agents || 0}`);
            outputChannel.appendLine(`   Tasks Completed: ${stats.lrs.tasks_completed || 0}`);
            outputChannel.appendLine(`   Avg Precision: ${stats.lrs.avg_precision?.toFixed(3) || 'N/A'}`);
            outputChannel.appendLine(`   Learning Events: ${stats.lrs.learning_events || 0}`);
            outputChannel.appendLine('');
        }

        // Benchmark results
        if (stats.benchmarks) {
            outputChannel.appendLine('🧪 Recent Benchmark Results:');
            if (stats.benchmarks.last_run) {
                outputChannel.appendLine(`   Last Run: ${new Date(stats.benchmarks.last_run * 1000).toLocaleString()}`);
            }
            outputChannel.appendLine(`   Chaos Success Rate: ${(stats.benchmarks.chaos_success_rate * 100)?.toFixed(1) || 'N/A'}%`);
            outputChannel.appendLine(`   GAIA Success Rate: ${(stats.benchmarks.gaia_success_rate * 100)?.toFixed(1) || 'N/A'}%`);
            outputChannel.appendLine('');
        }

        outputChannel.show();

        vscode.window.showInformationMessage('LRS System Statistics Loaded');
    });
}

async function benchmarkCommand(lrsClient: LRSClient): Promise<void> {
    const runBenchmarks = await vscode.window.showQuickPick([
        'Run Chaos Scriptorium Benchmark',
        'Run GAIA Benchmark',
        'Run Full Benchmark Suite',
        'Cancel'
    ], { placeHolder: 'Select benchmark to run' });

    if (!runBenchmarks || runBenchmarks === 'Cancel') {
        return;
    }

    await vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: `Running ${runBenchmarks}...`,
        cancellable: true
    }, async (progress, token) => {
        progress.report({ increment: 10, message: "Initializing benchmark..." });

        const benchmarkResults = await lrsClient.runBenchmarks();

        if (benchmarkResults.error) {
            vscode.window.showErrorMessage(`Benchmark failed: ${benchmarkResults.error}`);
            return;
        }

        progress.report({ increment: 90, message: "Processing results..." });

        // Create output channel for benchmark results
        const outputChannel = vscode.window.createOutputChannel('LRS Benchmark Results');
        outputChannel.clear();
        outputChannel.appendLine('🧪 LRS Benchmark Results');
        outputChannel.appendLine('=' * 25);
        outputChannel.appendLine(`📊 Benchmark: ${runBenchmarks}`);
        outputChannel.appendLine(`⏰ Execution Time: ${benchmarkResults.execution_time?.toFixed(2) || 'N/A'}s`);
        outputChannel.appendLine('');

        if (benchmarkResults.results) {
            benchmarkResults.results.forEach((result: any, index: number) => {
                outputChannel.appendLine(`📈 Test ${index + 1}: ${result.name || 'Unknown'}`);
                outputChannel.appendLine(`   Status: ${result.success ? '✅ Passed' : '❌ Failed'}`);
                outputChannel.appendLine(`   Duration: ${result.duration?.toFixed(3) || 'N/A'}s`);
                if (result.score !== undefined) {
                    outputChannel.appendLine(`   Score: ${result.score.toFixed(3)}`);
                }
                if (result.details) {
                    outputChannel.appendLine(`   Details: ${result.details}`);
                }
                outputChannel.appendLine('');
            });
        }

        // Summary
        const totalTests = benchmarkResults.results?.length || 0;
        const passedTests = benchmarkResults.results?.filter((r: any) => r.success).length || 0;
        const successRate = totalTests > 0 ? (passedTests / totalTests * 100) : 0;

        outputChannel.appendLine('📊 Summary:');
        outputChannel.appendLine(`   Total Tests: ${totalTests}`);
        outputChannel.appendLine(`   Passed: ${passedTests}`);
        outputChannel.appendLine(`   Success Rate: ${successRate.toFixed(1)}%`);
        outputChannel.appendLine(`   Average Score: ${benchmarkResults.avg_score?.toFixed(3) || 'N/A'}`);

        outputChannel.show();

        vscode.window.showInformationMessage(
            `Benchmark Complete: ${passedTests}/${totalTests} tests passed (${successRate.toFixed(1)}%)`
        );
    });
}

async function configureCommand(): Promise<void> {
    const config = vscode.workspace.getConfiguration('opencode.lrs');

    const action = await vscode.window.showQuickPick([
        'Edit Server URL',
        'Toggle Telemetry',
        'Toggle Auto-Analyze',
        'Change Precision Level',
        'Edit Supported Languages',
        'Reset to Defaults',
        'Cancel'
    ], { placeHolder: 'Select configuration option' });

    if (!action || action === 'Cancel') {
        return;
    }

    try {
        switch (action) {
            case 'Edit Server URL':
                const serverUrl = await vscode.window.showInputBox({
                    prompt: 'Enter LRS server URL',
                    value: config.get('serverUrl', 'http://localhost:8000')
                });
                if (serverUrl) {
                    await config.update('serverUrl', serverUrl, vscode.ConfigurationTarget.Global);
                    vscode.window.showInformationMessage('Server URL updated');
                }
                break;

            case 'Toggle Telemetry':
                const telemetry = config.get('enableTelemetry', true);
                await config.update('enableTelemetry', !telemetry, vscode.ConfigurationTarget.Global);
                vscode.window.showInformationMessage(`Telemetry ${!telemetry ? 'enabled' : 'disabled'}`);
                break;

            case 'Toggle Auto-Analyze':
                const autoAnalyze = config.get('autoAnalyze', false);
                await config.update('autoAnalyze', !autoAnalyze, vscode.ConfigurationTarget.Global);
                vscode.window.showInformationMessage(`Auto-analyze ${!autoAnalyze ? 'enabled' : 'disabled'}`);
                break;

            case 'Change Precision Level':
                const precisionLevel = await vscode.window.showQuickPick(
                    ['low', 'medium', 'high', 'adaptive'],
                    { placeHolder: 'Select precision level' }
                );
                if (precisionLevel) {
                    await config.update('precisionLevel', precisionLevel, vscode.ConfigurationTarget.Global);
                    vscode.window.showInformationMessage(`Precision level set to ${precisionLevel}`);
                }
                break;

            case 'Reset to Defaults':
                const confirm = await vscode.window.showQuickPick(
                    ['Yes, reset all settings', 'Cancel'],
                    { placeHolder: 'This will reset all LRS extension settings to defaults' }
                );
                if (confirm === 'Yes, reset all settings') {
                    await config.update('serverUrl', undefined, vscode.ConfigurationTarget.Global);
                    await config.update('enableTelemetry', undefined, vscode.ConfigurationTarget.Global);
                    await config.update('autoAnalyze', undefined, vscode.ConfigurationTarget.Global);
                    await config.update('precisionLevel', undefined, vscode.ConfigurationTarget.Global);
                    await config.update('supportedLanguages', undefined, vscode.ConfigurationTarget.Global);
                    vscode.window.showInformationMessage('All settings reset to defaults');
                }
                break;
        }
    } catch (error) {
        vscode.window.showErrorMessage(`Configuration update failed: ${error}`);
    }
}

async function autoAnalyzeDocument(document: vscode.TextDocument, lrsClient: LRSClient): Promise<void> {
    // Skip if file is too large
    const config = getLRSConfig();
    const fileSize = Buffer.byteLength(document.getText(), 'utf8');

    if (fileSize > config.maxFileSize) {
        return; // Skip large files
    }

    try {
        const result = await lrsClient.analyzeCode(document.getText(), document.languageId);
        if (result.success && result.recommendations && result.recommendations.length > 0) {
            // Show subtle notification
            const showDetails = await vscode.window.showInformationMessage(
                `LRS found ${result.recommendations.length} suggestions for ${document.fileName.split('/').pop()}`,
                'Show Details',
                'Dismiss'
            );

            if (showDetails === 'Show Details') {
                const outputChannel = vscode.window.createOutputChannel('LRS Auto-Analysis');
                outputChannel.clear();
                outputChannel.appendLine(`🤖 LRS Auto-Analysis: ${document.fileName}`);
                outputChannel.appendLine('=' * 40);
                result.recommendations.forEach((rec: string, index: number) => {
                    outputChannel.appendLine(`${index + 1}. ${rec}`);
                });
                outputChannel.show();
            }
        }
    } catch (error) {
        // Silently fail for auto-analysis to avoid interrupting workflow
        console.log('LRS auto-analysis failed:', error);
    }
}

class LRSCodelensProvider implements vscode.CodeActionProvider {
    public static readonly providedCodeActionKinds = [
        vscode.CodeActionKind.QuickFix,
        vscode.CodeActionKind.Refactor
    ];

    constructor(private lrsClient: LRSClient) {}

    provideCodeActions(
        document: vscode.TextDocument,
        range: vscode.Range,
        context: vscode.CodeActionContext,
        token: vscode.CancellationToken
    ): vscode.ProviderResult<vscode.CodeAction[]> {
        const actions: vscode.CodeAction[] = [];

        // Only provide actions for supported languages
        const config = getLRSConfig();
        if (!isLanguageSupported(document.languageId, config)) {
            return actions;
        }

        // LRS Analyze action
        const analyzeAction = new vscode.CodeAction(
            'Analyze with LRS',
            vscode.CodeActionKind.QuickFix
        );
        analyzeAction.command = {
            title: 'Analyze with LRS',
            command: 'opencode.lrs.analyze'
        };
        actions.push(analyzeAction);

        // LRS Refactor action (only if text is selected)
        if (!range.isEmpty) {
            const refactorAction = new vscode.CodeAction(
                'Refactor with LRS',
                vscode.CodeActionKind.Refactor
            );
            refactorAction.command = {
                title: 'Refactor with LRS',
                command: 'opencode.lrs.refactor'
            };
            actions.push(refactorAction);
        }

        return actions;
    }
}

async function updateStatusBar(statusBarItem: vscode.StatusBarItem, lrsClient: LRSClient): Promise<void> {
    try {
        const stats = await lrsClient.getStats();
        if (!stats.error) {
            const activeAgents = stats.lrs?.active_agents || 0;
            const avgPrecision = stats.lrs?.avg_precision || 0;

            statusBarItem.text = `$(robot) LRS: ${activeAgents} agents | ${(avgPrecision * 100).toFixed(1)}% precision`;
            statusBarItem.tooltip = `OpenCode LRS Integration\nActive Agents: ${activeAgents}\nAverage Precision: ${(avgPrecision * 100).toFixed(1)}%\nClick for detailed statistics`;
            statusBarItem.show();
        } else {
            statusBarItem.text = '$(robot) LRS: Disconnected';
            statusBarItem.tooltip = 'LRS server not available - Click for details';
            statusBarItem.show();
        }
    } catch (error) {
        statusBarItem.text = '$(robot) LRS: Error';
        statusBarItem.tooltip = 'LRS connection error';
        statusBarItem.show();
    }
}