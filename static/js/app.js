// 全局变量
let currentSelector = null;
let currentStocks = [];
let showDetails = false;

// 初始化
document.addEventListener('DOMContentLoaded', function() {
    loadSelectors();
    loadLatestDate();
});

// 加载选择器列表
async function loadSelectors() {
    try {
        const response = await fetch('/api/selectors');
        const selectors = await response.json();
        
        const container = document.getElementById('selector-buttons');
        container.innerHTML = '';
        
        selectors.forEach(selector => {
            const btn = document.createElement('button');
            btn.className = 'btn btn-primary';
            btn.textContent = selector.name;
            btn.onclick = () => runSelector(selector.name);
            container.appendChild(btn);
        });
    } catch (error) {
        console.error('加载选择器失败:', error);
    }
}

// 加载最新交易日
async function loadLatestDate() {
    try {
        const response = await fetch('/api/latest_date');
        const data = await response.json();
        
        if (data.success && data.date) {
            document.getElementById('latest-date').textContent = 
                `最新交易日: ${data.date}`;
        }
    } catch (error) {
        console.error('加载日期失败:', error);
    }
}

// 运行选择器
async function runSelector(selectorName) {
    currentSelector = selectorName;
    
    // 更新按钮状态
    document.querySelectorAll('.selector-buttons .btn').forEach(btn => {
        btn.classList.toggle('active', btn.textContent === selectorName);
    });
    
    // 显示加载状态
    document.getElementById('results-title').innerHTML = 
        `<span class="loading"></span> 正在运行: ${selectorName}`;
    document.getElementById('stock-grid').innerHTML = 
        '<div class="empty-state"><p>⏳ 加载中...</p></div>';
    
    try {
        const response = await fetch(`/api/select/${selectorName}`);
        const data = await response.json();
        
        if (data.success) {
            currentStocks = data.stocks;
            displayResults(selectorName, data);
        } else {
            alert('运行失败: ' + data.error);
        }
    } catch (error) {
        console.error('运行选择器失败:', error);
        alert('运行失败，请查看控制台');
    }
}

// 显示结果
function displayResults(selectorName, data) {
    // 更新标题
    document.getElementById('results-title').textContent = 
        `${selectorName} - 选股结果`;
    
    // 更新统计信息
    displayStats(data);
    
    // 显示股票列表
    displayStockGrid(data.stocks);
}

// 显示统计信息
function displayStats(data) {
    const statsHtml = `
        <div class="stat-item">
            <span class="stat-label">选中股票</span>
            <span class="stat-value">${data.count} 只</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">查询日期</span>
            <span class="stat-value">${data.date}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">平均涨跌</span>
            <span class="stat-value ${getChangeClass(calcAvgChange(data.stocks))}">
                ${calcAvgChange(data.stocks).toFixed(2)}%
            </span>
        </div>
    `;
    
    document.getElementById('stats').innerHTML = statsHtml;
}

// 计算平均涨跌幅
function calcAvgChange(stocks) {
    if (stocks.length === 0) return 0;
    const sum = stocks.reduce((acc, stock) => acc + stock.change, 0);
    return sum / stocks.length;
}

// 显示股票网格
function displayStockGrid(stocks) {
    const grid = document.getElementById('stock-grid');
    
    if (stocks.length === 0) {
        grid.innerHTML = '<div class="empty-state"><p>未找到符合条件的股票</p></div>';
        return;
    }
    
    grid.innerHTML = stocks.map(stock => createStockCard(stock)).join('');
}

// 创建股票卡片
function createStockCard(stock) {
    const changeClass = getChangeClass(stock.change);
    const changeSymbol = stock.change >= 0 ? '+' : '';
    
    return `
        <div class="stock-card" onclick="showChart('${stock.code}')">
            <div class="stock-code">${stock.code}</div>
            <div class="stock-price ${changeClass}">
                ¥${stock.close.toFixed(2)}
            </div>
            <div class="stock-change ${changeClass}">
                ${changeSymbol}${stock.change.toFixed(2)}%
            </div>
            <div class="stock-info">
                <div class="info-row">
                    <span>日期</span>
                    <span>${stock.date}</span>
                </div>
                <div class="info-row">
                    <span>成交量</span>
                    <span>${formatVolume(stock.volume)}</span>
                </div>
            </div>
            <div class="stock-details ${showDetails ? 'show' : ''}">
                <div class="info-row">
                    <span>开盘</span>
                    <span>¥${stock.open.toFixed(2)}</span>
                </div>
                <div class="info-row">
                    <span>最高</span>
                    <span>¥${stock.high.toFixed(2)}</span>
                </div>
                <div class="info-row">
                    <span>最低</span>
                    <span>¥${stock.low.toFixed(2)}</span>
                </div>
                ${stock.ma5 ? `
                <div class="info-row">
                    <span>MA5</span>
                    <span>¥${stock.ma5.toFixed(2)}</span>
                </div>
                ` : ''}
                ${stock.ma20 ? `
                <div class="info-row">
                    <span>MA20</span>
                    <span>¥${stock.ma20.toFixed(2)}</span>
                </div>
                ` : ''}
            </div>
        </div>
    `;
}

// 获取涨跌颜色类
function getChangeClass(change) {
    if (change > 0) return 'price-up';
    if (change < 0) return 'price-down';
    return 'price-neutral';
}

// 格式化成交量
function formatVolume(volume) {
    if (volume >= 100000000) {
        return (volume / 100000000).toFixed(2) + '亿';
    } else if (volume >= 10000) {
        return (volume / 10000).toFixed(2) + '万';
    }
    return volume.toFixed(0);
}

// 显示K线图
async function showChart(code, strategy = 'default') {
    const modal = document.getElementById('chart-modal');
    modal.classList.add('show');
    
    document.getElementById('modal-title').textContent = `${code} - K线图`;
    
    // 显示加载状态
    document.getElementById('kline-chart').innerHTML = 
        '<div style="text-align:center;padding:50px;">⏳ 加载中...</div>';
    
    // 添加战法选择器
    const strategyHtml = `
        <div style="padding: 10px; background: #f5f5f5; border-radius: 5px; margin-bottom: 10px;">
            <label style="font-weight: bold; margin-right: 10px;">信号类型：</label>
            <select id="strategy-selector" onchange="showChart('${code}', this.value)" style="padding: 5px 10px; border-radius: 3px; border: 1px solid #ddd;">
                <option value="default" ${strategy === 'default' ? 'selected' : ''}>少妇战法（B1/S1）</option>
                <option value="breakout" ${strategy === 'breakout' ? 'selected' : ''}>出坑战法（B/S）</option>
            </select>
            <span style="margin-left: 15px; color: #666; font-size: 12px;">
                ${strategy === 'default' ? 'B1=KDJ超卖转强, S1=放量大阴线' : 'B=接近前高突破, S=高位大阴线'}
            </span>
        </div>
    `;
    document.getElementById('backtest-result').innerHTML = strategyHtml;
    
    try {
        const response = await fetch(`/api/stock/${code}?days=120&strategy=${strategy}`);
        const data = await response.json();
        
        if (data.success) {
            // 使用联动绘制函数
            drawLinkedCharts(data);
        } else {
            alert('加载失败: ' + data.error);
        }
    } catch (error) {
        console.error('加载股票数据失败:', error);
        alert('加载失败，请查看控制台');
    }
}

// 绘制K线图（增强版：趋势线+交易信号）
function drawKlineChart(data) {
    // 计算价格范围，增加上下边距
    const prices = [...data.high, ...data.low];
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const priceRange = maxPrice - minPrice;
    const padding = priceRange * 0.15; // 上下各留15%空间（为信号标记留空间）
    
    // 准备K线数据
    const candlestick = {
        x: data.dates,
        open: data.open,
        high: data.high,
        low: data.low,
        close: data.close,
        type: 'candlestick',
        name: 'K线',
        increasing: {
            line: {color: '#ef5350', width: 1},
            fillcolor: '#ef5350'
        },
        decreasing: {
            line: {color: '#26a69a', width: 1},
            fillcolor: '#26a69a'
        },
        whiskerwidth: 0.5,
        xaxis: 'x',
        yaxis: 'y',
        hoverinfo: 'x+y'
    };
    
    // 趋势线（知行体系标准公式）
    const trendShort = {
        x: data.dates,
        y: data.trend_short,
        type: 'scatter',
        mode: 'lines',
        name: '短期趋势线',
        line: {color: '#FF6B6B', width: 2.5, dash: 'solid'},
        hovertemplate: '短期趋势: %{y:.2f}<extra></extra>'
    };
    
    const trendLong = {
        x: data.dates,
        y: data.trend_long,
        type: 'scatter',
        mode: 'lines',
        name: '知行多空线',
        line: {color: '#1976D2', width: 2.5, dash: 'solid'},
        hovertemplate: '知行多空线: %{y:.2f}<extra></extra>'
    };
    
    // 准备均线数据（辅助线，细一些）
    const ma5 = {
        x: data.dates,
        y: data.ma5,
        type: 'scatter',
        mode: 'lines',
        name: 'MA5',
        line: {color: '#FF6B6B', width: 1, dash: 'dot'},
        opacity: 0.5,
        hovertemplate: 'MA5: %{y:.2f}<extra></extra>'
    };
    
    const ma20 = {
        x: data.dates,
        y: data.ma20,
        type: 'scatter',
        mode: 'lines',
        name: 'MA20',
        line: {color: '#FFE66D', width: 1, dash: 'dot'},
        opacity: 0.5,
        hovertemplate: 'MA20: %{y:.2f}<extra></extra>'
    };
    
    const ma60 = {
        x: data.dates,
        y: data.ma60,
        type: 'scatter',
        mode: 'lines',
        name: 'MA60',
        line: {color: '#C7CEEA', width: 1, dash: 'dot'},
        opacity: 0.5,
        hovertemplate: 'MA60: %{y:.2f}<extra></extra>'
    };
    
    // 提取交易信号标记
    const b1Signals = (data.signals || []).filter(s => s.type === 'B1');
    const b2Signals = (data.signals || []).filter(s => s.type === 'B2');
    const s1Signals = (data.signals || []).filter(s => s.type === 'S1');
    
    // B1买点标记（绿色向上三角）
    const b1Markers = {
        x: b1Signals.map(s => s.date),
        y: b1Signals.map(s => s.price * 0.97), // 标记在价格下方3%
        mode: 'markers+text',
        type: 'scatter',
        name: 'B1买点',
        marker: {
            color: '#00C853',
            size: 15,
            symbol: 'triangle-up',
            line: {color: '#fff', width: 2}
        },
        text: b1Signals.map(() => 'B1'),
        textposition: 'bottom center',
        textfont: {color: '#00C853', size: 12, family: 'Arial Black'},
        hovertemplate: '<b>B1买点</b><br>价格: %{y:.2f}<br>%{x}<extra></extra>'
    };
    
    // B2加仓标记（深绿色向上三角）
    const b2Markers = {
        x: b2Signals.map(s => s.date),
        y: b2Signals.map(s => s.price * 0.97),
        mode: 'markers+text',
        type: 'scatter',
        name: 'B2加仓',
        marker: {
            color: '#1B5E20',
            size: 13,
            symbol: 'triangle-up',
            line: {color: '#fff', width: 2}
        },
        text: b2Signals.map(() => 'B2'),
        textposition: 'bottom center',
        textfont: {color: '#1B5E20', size: 11, family: 'Arial Black'},
        hovertemplate: '<b>B2加仓</b><br>价格: %{y:.2f}<br>%{x}<extra></extra>'
    };
    
    // S1卖点标记（红色向下三角）
    const s1Markers = {
        x: s1Signals.map(s => s.date),
        y: s1Signals.map(s => s.price * 1.03), // 标记在价格上方3%
        mode: 'markers+text',
        type: 'scatter',
        name: 'S1卖点',
        marker: {
            color: '#D32F2F',
            size: 15,
            symbol: 'triangle-down',
            line: {color: '#fff', width: 2}
        },
        text: s1Signals.map(() => 'S1'),
        textposition: 'top center',
        textfont: {color: '#D32F2F', size: 12, family: 'Arial Black'},
        hovertemplate: '<b>S1卖点</b><br>价格: %{y:.2f}<br>%{x}<extra></extra>'
    };
    
    const chartData = [
        candlestick, 
        trendShort, 
        trendLong,
        ma5, 
        ma20, 
        ma60
    ];
    
    // 添加交易信号标记
    if (b1Signals.length > 0) chartData.push(b1Markers);
    if (b2Signals.length > 0) chartData.push(b2Markers);
    if (s1Signals.length > 0) chartData.push(s1Markers);
    
    // 统计信号数量
    const signalSummary = `B1:${b1Signals.length} | B2:${b2Signals.length} | S1:${s1Signals.length}`;
    
    const layout = {
        title: {
            text: `${data.code} K线图 【${signalSummary}】`,
            font: {size: 18, color: '#333'}
        },
        xaxis: {
            rangeslider: {visible: false},
            type: 'date', // 使用date类型显示正确的日期
            showgrid: true,
            gridcolor: '#f0f0f0',
            tickformat: '%Y-%m-%d',
            tickangle: -45
        },
        yaxis: {
            title: '价格 (¥)',
            range: [minPrice - padding, maxPrice + padding],
            autorange: false,
            fixedrange: false,
            showgrid: true,
            gridcolor: '#f0f0f0'
        },
        showlegend: true,
        legend: {
            orientation: 'h',
            y: 1.15,
            x: 0.5,
            xanchor: 'center',
            bgcolor: 'rgba(255,255,255,0.9)',
            bordercolor: '#ddd',
            borderwidth: 1,
            font: {size: 11}
        },
        margin: {l: 60, r: 30, t: 90, b: 60},
        hovermode: 'x unified',
        plot_bgcolor: '#fafafa',
        paper_bgcolor: 'white',
        annotations: []
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    };
    
    Plotly.newPlot('kline-chart', chartData, layout, config);
}

// 绘制成交量图（优化版）
function drawVolumeChart(data) {
    // 计算颜色（根据涨跌）
    const colors = data.close.map((close, i) => {
        if (i === 0) return '#999';
        return close >= data.close[i-1] ? '#ef5350' : '#26a69a';
    });
    
    const volumeData = {
        x: data.dates,
        y: data.volume,
        type: 'bar',
        name: '成交量',
        marker: {
            color: colors,
            line: {width: 0}
        },
        hovertemplate: '成交量: %{y:.0f}<extra></extra>'
    };
    
    const layout = {
        title: {
            text: '成交量',
            font: {size: 14, color: '#666'}
        },
        xaxis: {
            type: 'date',
            showgrid: false,
            tickformat: '%Y-%m-%d',
            tickangle: -45
        },
        yaxis: {
            title: '量',
            showgrid: true,
            gridcolor: '#f0f0f0'
        },
        showlegend: false,
        margin: {l: 60, r: 30, t: 40, b: 40},
        hovermode: 'x unified',
        plot_bgcolor: '#fafafa',
        paper_bgcolor: 'white'
    };
    
    const config = {
        responsive: true,
        displayModeBar: false
    };
    
    Plotly.newPlot('volume-chart', [volumeData], layout, config);
}

// 绘制MACD图
function drawMACDChart(data) {
    // MACD柱状图颜色
    const macdColors = data.macd.map(val => val >= 0 ? '#ef5350' : '#26a69a');
    
    // MACD柱
    const macdBars = {
        x: data.dates,
        y: data.macd,
        type: 'bar',
        name: 'MACD',
        marker: {
            color: macdColors,
            line: {width: 0}
        },
        hovertemplate: 'MACD: %{y:.3f}<extra></extra>'
    };
    
    // DIF线
    const difLine = {
        x: data.dates,
        y: data.dif,
        type: 'scatter',
        mode: 'lines',
        name: 'DIF',
        line: {color: '#2196F3', width: 1.5},
        hovertemplate: 'DIF: %{y:.3f}<extra></extra>'
    };
    
    // DEA线
    const deaLine = {
        x: data.dates,
        y: data.dea,
        type: 'scatter',
        mode: 'lines',
        name: 'DEA',
        line: {color: '#FF9800', width: 1.5},
        hovertemplate: 'DEA: %{y:.3f}<extra></extra>'
    };
    
    // 零轴线
    const zeroLine = {
        x: data.dates,
        y: new Array(data.dates.length).fill(0),
        type: 'scatter',
        mode: 'lines',
        name: '',
        line: {color: '#999', width: 1, dash: 'dash'},
        showlegend: false,
        hoverinfo: 'skip'
    };
    
    const layout = {
        title: {
            text: 'MACD指标',
            font: {size: 14, color: '#666'}
        },
        xaxis: {
            type: 'date',
            showgrid: false,
            tickformat: '%Y-%m-%d',
            tickangle: -45
        },
        yaxis: {
            showgrid: true,
            gridcolor: '#f0f0f0',
            zeroline: false
        },
        showlegend: true,
        legend: {
            orientation: 'h',
            y: 1.15,
            x: 0.5,
            xanchor: 'center',
            font: {size: 10}
        },
        margin: {l: 60, r: 30, t: 50, b: 30},
        hovermode: 'x unified',
        plot_bgcolor: '#fafafa',
        paper_bgcolor: 'white'
    };
    
    const config = {
        responsive: true,
        displayModeBar: false
    };
    
    Plotly.newPlot('macd-chart', [zeroLine, macdBars, difLine, deaLine], layout, config);
}

// 绘制KDJ图
function drawKDJChart(data) {
    // K线
    const kLine = {
        x: data.dates,
        y: data.k,
        type: 'scatter',
        mode: 'lines',
        name: 'K',
        line: {color: '#2196F3', width: 1.5},
        hovertemplate: 'K: %{y:.2f}<extra></extra>'
    };
    
    // D线
    const dLine = {
        x: data.dates,
        y: data.d,
        type: 'scatter',
        mode: 'lines',
        name: 'D',
        line: {color: '#FF9800', width: 1.5},
        hovertemplate: 'D: %{y:.2f}<extra></extra>'
    };
    
    // J线
    const jLine = {
        x: data.dates,
        y: data.j,
        type: 'scatter',
        mode: 'lines',
        name: 'J',
        line: {color: '#9C27B0', width: 1.5},
        hovertemplate: 'J: %{y:.2f}<extra></extra>'
    };
    
    // 超买超卖线
    const overbought = {
        x: data.dates,
        y: new Array(data.dates.length).fill(80),
        type: 'scatter',
        mode: 'lines',
        name: '超买(80)',
        line: {color: '#f44336', width: 1, dash: 'dash'},
        hoverinfo: 'skip'
    };
    
    const oversold = {
        x: data.dates,
        y: new Array(data.dates.length).fill(20),
        type: 'scatter',
        mode: 'lines',
        name: '超卖(20)',
        line: {color: '#4caf50', width: 1, dash: 'dash'},
        hoverinfo: 'skip'
    };
    
    const layout = {
        title: {
            text: 'KDJ指标',
            font: {size: 14, color: '#666'}
        },
        xaxis: {
            type: 'date',
            showgrid: false,
            tickformat: '%Y-%m-%d',
            tickangle: -45
        },
        yaxis: {
            range: [0, 100],
            showgrid: true,
            gridcolor: '#f0f0f0'
        },
        showlegend: true,
        legend: {
            orientation: 'h',
            y: 1.15,
            x: 0.5,
            xanchor: 'center',
            font: {size: 10}
        },
        margin: {l: 60, r: 30, t: 50, b: 30},
        hovermode: 'x unified',
        plot_bgcolor: '#fafafa',
        paper_bgcolor: 'white'
    };
    
    const config = {
        responsive: true,
        displayModeBar: false
    };
    
    Plotly.newPlot('kdj-chart', [oversold, overbought, kLine, dLine, jLine], layout, config);
}

// 新增：联动绘制所有图表（支持缩放同步）
function drawLinkedCharts(data) {
    // 计算价格范围
    const prices = [...data.high, ...data.low];
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const priceRange = maxPrice - minPrice;
    const padding = priceRange * 0.15;
    
    // 提取交易信号（兼容B1/B和S1/S）
    const b1Signals = (data.signals || []).filter(s => s.type === 'B1' || s.type === 'B');
    const b2Signals = (data.signals || []).filter(s => s.type === 'B2');
    const s1Signals = (data.signals || []).filter(s => s.type === 'S1' || s.type === 'S');
    
    // === K线图数据 ===
    const candlestick = {
        x: data.dates,
        open: data.open,
        high: data.high,
        low: data.low,
        close: data.close,
        type: 'candlestick',
        name: 'K线',
        xaxis: 'x',
        yaxis: 'y',
        increasing: {line: {color: '#ef5350', width: 1}, fillcolor: '#ef5350'},
        decreasing: {line: {color: '#26a69a', width: 1}, fillcolor: '#26a69a'}
    };
    
    const trendShort = {
        x: data.dates,
        y: data.trend_short,
        type: 'scatter',
        mode: 'lines',
        name: '短期趋势线',
        xaxis: 'x',
        yaxis: 'y',
        line: {color: '#FF6B6B', width: 2.5}
    };
    
    const trendLong = {
        x: data.dates,
        y: data.trend_long,
        type: 'scatter',
        mode: 'lines',
        name: '知行多空线',
        xaxis: 'x',
        yaxis: 'y',
        line: {color: '#1976D2', width: 2.5}
    };
    
    // B1/B2/S1标记
    const traces = [candlestick, trendShort, trendLong];
    
    if (b1Signals.length > 0) {
        traces.push({
            x: b1Signals.map(s => s.date),
            y: b1Signals.map(s => s.price * 0.97),
            mode: 'markers+text',
            type: 'scatter',
            name: b1Signals[0].type === 'B1' ? 'B1买点' : 'B买点',
            xaxis: 'x',
            yaxis: 'y',
            marker: {color: '#00C853', size: 15, symbol: 'triangle-up', line: {color: '#fff', width: 2}},
            text: b1Signals.map(s => s.type),
            textposition: 'bottom center',
            textfont: {color: '#00C853', size: 12}
        });
    }
    
    if (b2Signals.length > 0) {
        traces.push({
            x: b2Signals.map(s => s.date),
            y: b2Signals.map(s => s.price * 0.97),
            mode: 'markers+text',
            type: 'scatter',
            name: 'B2加仓',
            xaxis: 'x',
            yaxis: 'y',
            marker: {color: '#1B5E20', size: 13, symbol: 'triangle-up', line: {color: '#fff', width: 2}},
            text: b2Signals.map(() => 'B2'),
            textposition: 'bottom center',
            textfont: {color: '#1B5E20', size: 11}
        });
    }
    
    if (s1Signals.length > 0) {
        traces.push({
            x: s1Signals.map(s => s.date),
            y: s1Signals.map(s => s.price * 1.03),
            mode: 'markers+text',
            type: 'scatter',
            name: s1Signals[0].type === 'S1' ? 'S1卖点' : 'S卖点',
            xaxis: 'x',
            yaxis: 'y',
            marker: {color: '#D32F2F', size: 15, symbol: 'triangle-down', line: {color: '#fff', width: 2}},
            text: s1Signals.map(s => s.type),
            textposition: 'top center',
            textfont: {color: '#D32F2F', size: 12}
        });
    }
    
    // === 成交量数据 ===
    const colors = data.close.map((close, i) => {
        if (i === 0) return '#999';
        return close >= data.close[i-1] ? '#ef5350' : '#26a69a';
    });
    
    traces.push({
        x: data.dates,
        y: data.volume,
        type: 'bar',
        name: '成交量',
        xaxis: 'x2',
        yaxis: 'y2',
        marker: {color: colors, line: {width: 0}}
    });
    
    // === MACD数据 ===
    const macdColors = data.macd.map(val => val >= 0 ? '#ef5350' : '#26a69a');
    
    traces.push({
        x: data.dates,
        y: new Array(data.dates.length).fill(0),
        type: 'scatter',
        mode: 'lines',
        name: '',
        xaxis: 'x3',
        yaxis: 'y3',
        line: {color: '#999', width: 1, dash: 'dash'},
        showlegend: false,
        hoverinfo: 'skip'
    });
    
    traces.push({
        x: data.dates,
        y: data.macd,
        type: 'bar',
        name: 'MACD',
        xaxis: 'x3',
        yaxis: 'y3',
        marker: {color: macdColors, line: {width: 0}}
    });
    
    traces.push({
        x: data.dates,
        y: data.dif,
        type: 'scatter',
        mode: 'lines',
        name: 'DIF',
        xaxis: 'x3',
        yaxis: 'y3',
        line: {color: '#2196F3', width: 1.5}
    });
    
    traces.push({
        x: data.dates,
        y: data.dea,
        type: 'scatter',
        mode: 'lines',
        name: 'DEA',
        xaxis: 'x3',
        yaxis: 'y3',
        line: {color: '#FF9800', width: 1.5}
    });
    
    // === KDJ数据 ===
    traces.push({
        x: data.dates,
        y: new Array(data.dates.length).fill(80),
        type: 'scatter',
        mode: 'lines',
        name: '超买(80)',
        xaxis: 'x4',
        yaxis: 'y4',
        line: {color: '#f44336', width: 1, dash: 'dash'},
        hoverinfo: 'skip'
    });
    
    traces.push({
        x: data.dates,
        y: new Array(data.dates.length).fill(20),
        type: 'scatter',
        mode: 'lines',
        name: '超卖(20)',
        xaxis: 'x4',
        yaxis: 'y4',
        line: {color: '#4caf50', width: 1, dash: 'dash'},
        hoverinfo: 'skip'
    });
    
    traces.push({
        x: data.dates,
        y: data.k,
        type: 'scatter',
        mode: 'lines',
        name: 'K',
        xaxis: 'x4',
        yaxis: 'y4',
        line: {color: '#2196F3', width: 1.5}
    });
    
    traces.push({
        x: data.dates,
        y: data.d,
        type: 'scatter',
        mode: 'lines',
        name: 'D',
        xaxis: 'x4',
        yaxis: 'y4',
        line: {color: '#FF9800', width: 1.5}
    });
    
    traces.push({
        x: data.dates,
        y: data.j,
        type: 'scatter',
        mode: 'lines',
        name: 'J',
        xaxis: 'x4',
        yaxis: 'y4',
        line: {color: '#9C27B0', width: 1.5}
    });
    
    // === 布局配置（4个子图，共享X轴实现联动） ===
    const signalSummary = `B1:${b1Signals.length} | B2:${b2Signals.length} | S1:${s1Signals.length}`;
    
    const layout = {
        title: {
            text: `${data.code} K线图 【${signalSummary}】`,
            font: {size: 18, color: '#333'}
        },
        grid: {
            rows: 4,
            columns: 1,
            pattern: 'independent',
            roworder: 'top to bottom',
            subplots: [['xy'], ['x2y2'], ['x3y3'], ['x4y4']]
        },
        // K线图（主图）
        xaxis: {
            type: 'date',
            rangeslider: {visible: false},
            showticklabels: false,
            matches: 'x2'  // 与x2联动
        },
        yaxis: {
            title: '价格 (¥)',
            domain: [0.55, 1],
            range: [minPrice - padding, maxPrice + padding],
            autorange: false
        },
        // 成交量图
        xaxis2: {
            type: 'date',
            showticklabels: false,
            matches: 'x3'  // 与x3联动
        },
        yaxis2: {
            title: '量',
            domain: [0.4, 0.52]
        },
        // MACD图
        xaxis3: {
            type: 'date',
            showticklabels: false,
            matches: 'x4'  // 与x4联动
        },
        yaxis3: {
            title: 'MACD',
            domain: [0.2, 0.37]
        },
        // KDJ图
        xaxis4: {
            type: 'date',
            tickformat: '%Y-%m-%d',
            tickangle: -45
        },
        yaxis4: {
            title: 'KDJ',
            domain: [0, 0.17],
            range: [0, 100]
        },
        showlegend: true,
        legend: {
            orientation: 'h',
            y: 1.08,
            x: 0.5,
            xanchor: 'center',
            font: {size: 10}
        },
        margin: {l: 60, r: 30, t: 100, b: 60},
        hovermode: 'x unified',
        plot_bgcolor: '#fafafa',
        paper_bgcolor: 'white',
        height: 900
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['lasso2d', 'select2d']
    };
    
    // 清空现有图表
    document.getElementById('kline-chart').innerHTML = '';
    document.getElementById('volume-chart').innerHTML = '';
    document.getElementById('macd-chart').innerHTML = '';
    document.getElementById('kdj-chart').innerHTML = '';
    
    // 绘制到K线图容器
    Plotly.newPlot('kline-chart', traces, layout, config);
    
    // 显示回测结果（追加到战法选择器后面）
    if (data.backtest) {
        const currentHtml = document.getElementById('backtest-result').innerHTML;
        const backtestHtml = generateBacktestHTML(data.backtest, data.code);
        document.getElementById('backtest-result').innerHTML = currentHtml + backtestHtml;
    }
}

// 生成回测HTML
function generateBacktestHTML(backtest, code) {
    
    if (backtest.total_trades === 0) {
        return `
            <div style="background: #fff3cd; padding: 15px; border-radius: 5px; border-left: 4px solid #ffc107;">
                <h4 style="margin: 0 0 10px 0;">📊 回测结果</h4>
                <p style="margin: 0; color: #856404;">暂无完整交易（需要买入和卖出配对）</p>
            </div>
        `;
    }
    
    const winRateColor = backtest.win_rate >= 60 ? '#4caf50' : (backtest.win_rate >= 40 ? '#ff9800' : '#f44336');
    const totalReturnColor = backtest.total_return >= 0 ? '#4caf50' : '#f44336';
    
    let tradesHtml = '';
    backtest.trades.forEach((trade, idx) => {
        const returnColor = trade.return_pct >= 0 ? '#4caf50' : '#f44336';
        const statusBadge = trade.status === 'open' ? '<span style="background:#ff9800;color:white;padding:2px 6px;border-radius:3px;font-size:11px;">持仓中</span>' : '';
        tradesHtml += `
            <tr>
                <td>${idx + 1}</td>
                <td>${trade.buy_date}</td>
                <td>¥${trade.buy_price.toFixed(2)}</td>
                <td>${trade.sell_date}</td>
                <td>¥${trade.sell_price.toFixed(2)}</td>
                <td style="color: ${returnColor}; font-weight: bold;">${trade.return_pct > 0 ? '+' : ''}${trade.return_pct}%</td>
                <td>${trade.days_held}天 ${statusBadge}</td>
            </tr>
        `;
    });
    
    return `
        <div style="background: white; padding: 20px; border-radius: 8px; border: 1px solid #e0e0e0; margin-top: 20px;">
            <h3 style="margin: 0 0 15px 0; color: #333; border-bottom: 2px solid #1976D2; padding-bottom: 10px;">
                📊 ${code} 回测报告（B1买入 → S1卖出）
            </h3>
            
            <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 15px; margin-bottom: 20px;">
                <div style="background: #f5f5f5; padding: 15px; border-radius: 5px; text-align: center;">
                    <div style="font-size: 12px; color: #666;">交易次数</div>
                    <div style="font-size: 24px; font-weight: bold; color: #333; margin: 5px 0;">${backtest.total_trades}</div>
                </div>
                <div style="background: #f5f5f5; padding: 15px; border-radius: 5px; text-align: center;">
                    <div style="font-size: 12px; color: #666;">胜率</div>
                    <div style="font-size: 24px; font-weight: bold; color: ${winRateColor}; margin: 5px 0;">${backtest.win_rate}%</div>
                    <div style="font-size: 11px; color: #999;">${backtest.win_count}胜 ${backtest.loss_count}负</div>
                </div>
                <div style="background: #f5f5f5; padding: 15px; border-radius: 5px; text-align: center;">
                    <div style="font-size: 12px; color: #666;">平均收益</div>
                    <div style="font-size: 24px; font-weight: bold; color: ${backtest.avg_return >= 0 ? '#4caf50' : '#f44336'}; margin: 5px 0;">
                        ${backtest.avg_return > 0 ? '+' : ''}${backtest.avg_return}%
                    </div>
                </div>
                <div style="background: #f5f5f5; padding: 15px; border-radius: 5px; text-align: center;">
                    <div style="font-size: 12px; color: #666;">累计收益</div>
                    <div style="font-size: 24px; font-weight: bold; color: ${totalReturnColor}; margin: 5px 0;">
                        ${backtest.total_return > 0 ? '+' : ''}${backtest.total_return}%
                    </div>
                </div>
                <div style="background: #f5f5f5; padding: 15px; border-radius: 5px; text-align: center;">
                    <div style="font-size: 12px; color: #666;">最大回撤</div>
                    <div style="font-size: 24px; font-weight: bold; color: #f44336; margin: 5px 0;">-${backtest.max_drawdown}%</div>
                </div>
            </div>
            
            <details open>
                <summary style="cursor: pointer; font-weight: bold; color: #1976D2; margin-bottom: 10px;">
                    📋 交易明细（${backtest.total_trades}笔）
                </summary>
                <div style="overflow-x: auto;">
                    <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
                        <thead>
                            <tr style="background: #f5f5f5;">
                                <th style="padding: 10px; text-align: left; border-bottom: 2px solid #ddd;">#</th>
                                <th style="padding: 10px; text-align: left; border-bottom: 2px solid #ddd;">买入日期</th>
                                <th style="padding: 10px; text-align: left; border-bottom: 2px solid #ddd;">买入价</th>
                                <th style="padding: 10px; text-align: left; border-bottom: 2px solid #ddd;">卖出日期</th>
                                <th style="padding: 10px; text-align: left; border-bottom: 2px solid #ddd;">卖出价</th>
                                <th style="padding: 10px; text-align: left; border-bottom: 2px solid #ddd;">收益率</th>
                                <th style="padding: 10px; text-align: left; border-bottom: 2px solid #ddd;">持有</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${tradesHtml}
                        </tbody>
                    </table>
                </div>
            </details>
        </div>
    `;
}

// 关闭模态框
function closeModal() {
    document.getElementById('chart-modal').classList.remove('show');
}

// 排序股票
function sortStocks() {
    const sortBy = document.getElementById('sort-by').value;
    
    currentStocks.sort((a, b) => {
        switch(sortBy) {
            case 'code':
                return a.code.localeCompare(b.code);
            case 'change':
                return b.change - a.change;
            case 'volume':
                return b.volume - a.volume;
            default:
                return 0;
        }
    });
    
    displayStockGrid(currentStocks);
}

// 切换详情显示
function toggleDetails() {
    showDetails = document.getElementById('show-details').checked;
    displayStockGrid(currentStocks);
}

// 对比所有战法
async function compareAll() {
    try {
        const response = await fetch('/api/compare');
        const data = await response.json();
        
        if (data.success) {
            displayCompareResults(data);
        } else {
            alert('对比失败: ' + data.error);
        }
    } catch (error) {
        console.error('对比失败:', error);
        alert('对比失败，请查看控制台');
    }
}

// 显示对比结果
function displayCompareResults(data) {
    const modal = document.getElementById('compare-modal');
    modal.classList.add('show');
    
    let html = `
        <table class="compare-table">
            <thead>
                <tr>
                    <th>战法</th>
                    <th>选中数量</th>
                    <th>股票代码</th>
                </tr>
            </thead>
            <tbody>
    `;
    
    data.results.forEach(result => {
        const codes = result.codes.slice(0, 20).map(code => 
            `<span class="stock-tag" onclick="showChart('${code}')">${code}</span>`
        ).join('');
        
        const moreText = result.codes.length > 20 ? 
            `<span style="color:#999">...等${result.codes.length}只</span>` : '';
        
        html += `
            <tr>
                <td><strong>${result.name}</strong></td>
                <td>${result.count} 只</td>
                <td>
                    <div class="stock-list">
                        ${codes}
                        ${moreText}
                    </div>
                </td>
            </tr>
        `;
    });
    
    html += `
            </tbody>
        </table>
    `;
    
    document.getElementById('compare-results').innerHTML = html;
}

// 关闭对比模态框
function closeCompareModal() {
    document.getElementById('compare-modal').classList.remove('show');
}

// 点击模态框背景关闭
window.onclick = function(event) {
    const chartModal = document.getElementById('chart-modal');
    const compareModal = document.getElementById('compare-modal');
    
    if (event.target === chartModal) {
        closeModal();
    }
    if (event.target === compareModal) {
        closeCompareModal();
    }
}

