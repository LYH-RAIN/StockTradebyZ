// 增强版K线图绘制函数

// 绘制K线图（优化版）
function drawKlineChart(data) {
    // 计算价格范围，增加上下边距
    const prices = [...data.high, ...data.low];
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const priceRange = maxPrice - minPrice;
    const padding = priceRange * 0.1; // 上下各留10%空间
    
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
        hoverinfo: 'x+y'
    };
    
    // 准备均线数据
    const ma5 = {
        x: data.dates,
        y: data.ma5,
        type: 'scatter',
        mode: 'lines',
        name: 'MA5',
        line: {color: '#FF6B6B', width: 1.5},
        hovertemplate: 'MA5: %{y:.2f}<extra></extra>'
    };
    
    const ma10 = {
        x: data.dates,
        y: data.ma10,
        type: 'scatter',
        mode: 'lines',
        name: 'MA10',
        line: {color: '#4ECDC4', width: 1.5},
        hovertemplate: 'MA10: %{y:.2f}<extra></extra>'
    };
    
    const ma20 = {
        x: data.dates,
        y: data.ma20,
        type: 'scatter',
        mode: 'lines',
        name: 'MA20',
        line: {color: '#FFE66D', width: 1.5},
        hovertemplate: 'MA20: %{y:.2f}<extra></extra>'
    };
    
    const ma30 = {
        x: data.dates,
        y: data.ma30,
        type: 'scatter',
        mode: 'lines',
        name: 'MA30',
        line: {color: '#95E1D3', width: 1},
        hovertemplate: 'MA30: %{y:.2f}<extra></extra>'
    };
    
    const ma60 = {
        x: data.dates,
        y: data.ma60,
        type: 'scatter',
        mode: 'lines',
        name: 'MA60',
        line: {color: '#C7CEEA', width: 1},
        hovertemplate: 'MA60: %{y:.2f}<extra></extra>'
    };
    
    const chartData = [candlestick, ma5, ma10, ma20, ma30, ma60];
    
    const layout = {
        title: {
            text: `${data.code} K线图`,
            font: {size: 18, color: '#333'}
        },
        xaxis: {
            rangeslider: {visible: false},
            type: 'category', // 使用category类型自动去掉周末和节假日
            showgrid: true,
            gridcolor: '#f0f0f0'
        },
        yaxis: {
            title: '价格 (¥)',
            range: [minPrice - padding, maxPrice + padding], // 设置Y轴范围
            autorange: false, // 禁用自动范围
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
            bgcolor: 'rgba(255,255,255,0.8)',
            bordercolor: '#ddd',
            borderwidth: 1
        },
        margin: {l: 60, r: 30, t: 80, b: 40},
        hovermode: 'x unified',
        plot_bgcolor: '#fafafa',
        paper_bgcolor: 'white'
    };
    
    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        toImageButtonOptions: {
            format: 'png',
            filename: `${data.code}_kline`,
            height: 600,
            width: 1200,
            scale: 2
        }
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
            type: 'category',
            showgrid: false
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
            type: 'category',
            showgrid: false
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
            type: 'category',
            showgrid: false
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

