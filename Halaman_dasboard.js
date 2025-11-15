// =========================================================
// GLOBAL VARIABLES
// =========================================================
let interviewData = null;

// =========================================================
// DATA LOADING
// =========================================================
async function loadJSONData() {
    try {
        const response = await fetch('Halaman_dasboard.json');
        if (!response.ok) throw new Error('Failed to load data');
        
        interviewData = await response.json();
        console.log('✅ Data berhasil dimuat:', interviewData);
        loadDashboardData();
    } catch (error) {
        console.error('❌ Error loading JSON:', error);
        alert('Gagal memuat data. Pastikan file Halaman_dasboard.json tersedia.');
    }
}

// =========================================================
// CALCULATE AGGREGATE DATA
// =========================================================
function calculateAggregateData() {
    if (!interviewData?.content || interviewData.content.length === 0) {
        console.warn('⚠️ Data content kosong');
        return null;
    }

    const totalQuestions = interviewData.content.length;
    const aggregate = {
        avgConfidence: 0,
        avgKualitas: 0,
        avgRelevansi: 0,
        avgKoherensi: 0,
        avgTempo: 0,
        avgTotal: 0
    };

    // Hitung total dari semua pertanyaan
    interviewData.content.forEach(item => {
        const p = item.result.penilaian;
        aggregate.avgConfidence += p.confidence_score;
        aggregate.avgKualitas += p.kualitas_jawaban;
        aggregate.avgRelevansi += p.relevansi;
        aggregate.avgKoherensi += p.koherensi;
        aggregate.avgTempo += p.tempo_bicara;
        aggregate.avgTotal += p.total;
    });

    // Hitung rata-rata
    aggregate.avgConfidence = Math.round(aggregate.avgConfidence / totalQuestions);
    aggregate.avgKualitas = Math.round(aggregate.avgKualitas / totalQuestions);
    aggregate.avgRelevansi = Math.round(aggregate.avgRelevansi / totalQuestions);
    aggregate.avgKoherensi = Math.round(aggregate.avgKoherensi / totalQuestions);
    aggregate.avgTempo = Math.round(aggregate.avgTempo / totalQuestions);
    aggregate.avgTotal = Math.round(aggregate.avgTotal / totalQuestions);

    return aggregate;
}

// =========================================================
// MAIN DASHBOARD LOADER
// =========================================================
function loadDashboardData() {
    if (!interviewData) {
        console.error('❌ Data interview belum dimuat!');
        return;
    }
    
    updateAspectDetails();
    updateSummaryCards();
    updateCheatingDisplay();
    updateNonVerbalDisplay();
    updateFinalDecision();
    updateFinalRating();
    createRadarChart();
    
    console.log('✅ Dashboard berhasil dimuat');
}

// =========================================================
// UPDATE FUNCTIONS
// =========================================================
function updateCheatingDisplay() {
    const cheatingElement = document.getElementById('cheating-detect');
    if (!cheatingElement || !interviewData?.content) return;

    const result = interviewData.content[0].result;
    
    if (result.cheating_detection.toLowerCase() === 'ya') {
        cheatingElement.innerHTML = `
            <div class="content-text">
                YA
                <p style="font-size: 12px; margin-top: 5px; color: #e74c3c;">
                    Alasan: ${result.alasan_cheating || 'Tidak ada alasan'}
                </p>
            </div>
        `;
    } else {
        cheatingElement.innerHTML = '<div class="content-text">TIDAK</div>';
    }
}

function updateNonVerbalDisplay() {
    const nonVerbalElement = document.getElementById('nonverbal-analysis');
    if (!nonVerbalElement) return;
    
    const lastIndex = interviewData.content.length - 1;
    const analisisText = interviewData.content[lastIndex].result.analisis_non_verbal;
    
    nonVerbalElement.textContent = analisisText;
}

function updateFinalDecision() {
    const decisionElement = document.getElementById('final-decision');
    if (!decisionElement || !interviewData?.content) return;

    const result = interviewData.content[0].result;
    decisionElement.innerHTML = `<div class="content-text">${result.keputusan_akhir}</div>`;
}

function updateFinalRating() {
    const scoreImageElement = document.getElementById('scoreImage');
    if (!scoreImageElement) return;
    
    const rating = interviewData.content[0].result.penilaian_akhir;
    scoreImageElement.src = `Assest/rating-dark-${rating}.png`;
    scoreImageElement.alt = `Rating ${rating} dari 5`;
}

function updateAspectDetails() {
    const detailsList = document.getElementById('aspectDetailsList');
    if (!detailsList) return;
    
    const aggregate = calculateAggregateData();
    const aspects = [
        { label: 'Confidence Score', score: aggregate.avgConfidence },
        { label: 'Kualitas Jawaban', score: aggregate.avgKualitas },
        { label: 'Relevansi', score: aggregate.avgRelevansi },
        { label: 'Koherensi', score: aggregate.avgKoherensi },
        { label: 'Tempo Bicara', score: aggregate.avgTempo }
    ];
    
    detailsList.innerHTML = aspects.map(aspect => `
        <div class="aspect-detail-item">
            <div class="aspect-label">${aspect.label}</div>
            <div class="aspect-bar-container">
                <div class="aspect-progress-bar">
                    <div class="aspect-progress-fill" style="width: ${aspect.score}%"></div>
                </div>
                <div class="aspect-scale">
                    <span>0</span>
                    <span>50</span>
                    <span>100</span>
                </div>
            </div>
            <div class="aspect-score">${aspect.score}</div>
        </div>
    `).join('');
}

function updateSummaryCards() {
    const aggregate = calculateAggregateData();
    const aspects = [
        { label: 'Confidence Score', score: aggregate.avgConfidence },
        { label: 'Kualitas Jawaban', score: aggregate.avgKualitas },
        { label: 'Relevansi', score: aggregate.avgRelevansi },
        { label: 'Koherensi', score: aggregate.avgKoherensi },
        { label: 'Tempo Bicara', score: aggregate.avgTempo }
    ];
    
    // Update rata-rata skor total
    document.getElementById('averageScore').textContent = aggregate.avgTotal;
    
    // Update aspek tertinggi
    const maxScore = Math.max(...aspects.map(a => a.score));
    const maxAspect = aspects.find(a => a.score === maxScore);
    document.getElementById('highestAspect').textContent = maxAspect.label;
    document.getElementById('highestScore').textContent = maxScore;
    
    // Update aspek terendah
    const minScore = Math.min(...aspects.map(a => a.score));
    const minAspect = aspects.find(a => a.score === minScore);
    document.getElementById('lowestAspect').textContent = minAspect.label;
    document.getElementById('lowestScore').textContent = minScore;
    
    // Update konsistensi (Standard Deviation)
    const scores = aspects.map(a => a.score);
    const average = scores.reduce((a, b) => a + b, 0) / scores.length;
    const variance = scores.reduce((sum, score) => sum + Math.pow(score - average, 2), 0) / scores.length;
    const stdDev = Math.round(Math.sqrt(variance));
    document.getElementById('consistencyScore').textContent = `±${stdDev}`;
}

// =========================================================
// RADAR CHART
// =========================================================
function createRadarChart() {
    const ctx = document.getElementById('radarChart');
    if (!ctx) return;

    const aggregate = calculateAggregateData();
    const chartData = [
        aggregate.avgConfidence,
        aggregate.avgKualitas,
        aggregate.avgRelevansi,
        aggregate.avgKoherensi,
        aggregate.avgTempo
    ];
    
    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Confidence', 'Kualitas', 'Relevansi', 'Koherensi', 'Tempo'],
            datasets: [{
                label: 'Skor Kandidat',
                data: chartData,
                backgroundColor: 'rgba(102, 126, 234, 0.2)',
                borderColor: '#667eea',
                borderWidth: 2,
                pointBackgroundColor: '#667eea',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: '#667eea'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    ticks: { stepSize: 20 }
                }
            },
            plugins: { 
                legend: { position: 'top' } 
            }
        }
    });
}

// =========================================================
// UTILITY FUNCTIONS
// =========================================================

function downloadJSON() {
    if (!interviewData) {
        alert('Data belum dimuat!');
        return;
    }

    // Konversi data ke JSON string dengan format yang rapi
    const jsonString = JSON.stringify(interviewData, null, 2);
    
    // Buat blob dari JSON string
    const blob = new Blob([jsonString], { type: 'application/json' });
    
    // Buat URL untuk blob
    const url = URL.createObjectURL(blob);
    
    // Buat element anchor untuk download
    const a = document.createElement('a');
    a.href = url;
    a.download = `interview_data_${new Date().getTime()}.json`;
    
    // Trigger download
    document.body.appendChild(a);
    a.click();
    
    // Cleanup
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    console.log('✅ JSON berhasil didownload');
}


async function downloadPDF() {
    if (!interviewData) {
        alert('Data belum dimuat!');
        return;
    }

    // Import jsPDF
    const script = document.createElement('script');
    script.src = 'https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js';
    document.head.appendChild(script);

    script.onload = function () {
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF();

        // 🎨 Tema warna
        const blue = [0, 82, 211];      // #0052d3
        const cream = [255, 243, 217];  // #fff3d9

        // 📐 Layout
        const tableX = 20;
        const tableWidth = 170;
        const result = interviewData.content[0].result;
        const aggregate = calculateAggregateData();
        let yPos = 10;

        // ===== HEADER =====
        doc.setFontSize(20);
        doc.setTextColor(...blue);
        doc.text('LAPORAN ANALISIS INTERVIEW', 105, yPos, { align: 'center' });
        yPos += 15;
        doc.setDrawColor(...blue);
        doc.line(20, yPos, 190, yPos);
        yPos += 10;

        // ===== RINGKASAN PENILAIAN =====
        doc.setFontSize(14);
        doc.setTextColor(...blue);
        doc.text('Ringkasan Penilaian', 20, yPos);
        yPos += 5;

        doc.setFontSize(10);
        doc.setTextColor(0, 0, 0);
        doc.text(`Rata-rata Skor: ${aggregate.avgTotal}`, 20, yPos);
        doc.text(`Penilaian Akhir: ${result.penilaian_akhir}/5`, 80, yPos);
        doc.text(`Keputusan: ${result.keputusan_akhir}`, 140, yPos);
        yPos += 10;

        // ===== SKOR PER ASPEK =====
        yPos = drawTableHeader(doc, 'Skor Per Aspek Kompetensi', yPos, tableX, blue);
        const aspects = [
            ['Confidence Score', aggregate.avgConfidence, getScoreCategoryText(aggregate.avgConfidence)],
            ['Kualitas Jawaban', aggregate.avgKualitas, getScoreCategoryText(aggregate.avgKualitas)],
            ['Relevansi', aggregate.avgRelevansi, getScoreCategoryText(aggregate.avgRelevansi)],
            ['Koherensi', aggregate.avgKoherensi, getScoreCategoryText(aggregate.avgKoherensi)],
            ['Tempo Bicara', aggregate.avgTempo, getScoreCategoryText(aggregate.avgTempo)]
        ];
        yPos = drawTable(doc, aspects, ['Aspek Penilaian', 'Skor', 'Kategori'], yPos, tableX, tableWidth, blue, cream);

        // ===== HASIL ANALISIS =====
        yPos = drawTableHeader(doc, 'Hasil Analisis', yPos + 10, tableX, blue);
        const analysisData = [['Cheating Detection', result.cheating_detection || 'Tidak']];
        if (result.alasan_cheating) {
            analysisData.push(['Alasan Cheating', result.alasan_cheating]);
        }
        analysisData.push(['Analisis Non-Verbal', result.analisis_non_verbal]);
        yPos = drawTable(doc, analysisData, ['Kategori', 'Hasil'], yPos, tableX, tableWidth, blue, cream);

        // ===== DETAIL PERTANYAAN =====
        yPos += 10;
        if (yPos > 240) {
            doc.addPage();
            yPos = 25;
        }
        doc.setFontSize(12);
        doc.setTextColor(...blue);
        doc.text(`Detail Pertanyaan (${interviewData.content.length} Pertanyaan)`, 20, yPos);
        yPos += 7.5;

        interviewData.content.forEach((item, idx) => {
            if (yPos > 240) {
                doc.addPage();
                yPos = 25;
            }

            doc.setFontSize(10);
            doc.setTextColor(45, 55, 72);
            doc.text(`Pertanyaan ${idx + 1}:`, 20, yPos);
            yPos += 5;

            doc.setFontSize(9);
            const qLines = doc.splitTextToSize(item.question, 170);
            doc.text(qLines, 20, yPos);
            yPos += qLines.length * 5 + 5;

            const scores = [
                ['Confidence Score', item.result.penilaian.confidence_score],
                ['Kualitas Jawaban', item.result.penilaian.kualitas_jawaban],
                ['Relevansi', item.result.penilaian.relevansi],
                ['Koherensi', item.result.penilaian.koherensi],
                ['Tempo Bicara', item.result.penilaian.tempo_bicara],
                ['Total Skor', item.result.penilaian.total]
            ];
            yPos = drawTable(doc, scores, ['Aspek', 'Nilai'], yPos, tableX, tableWidth, blue, cream);
            yPos += 8;
        });

        // ===== FOOTER =====
        const pageCount = doc.internal.getNumberOfPages();
        for (let i = 1; i <= pageCount; i++) {
            doc.setPage(i);
            doc.setFontSize(8);
            doc.setTextColor(80, 80, 80);
            doc.text('Laporan ini dibuat secara otomatis oleh AI Interview Platform', 105, 285, { align: 'center' });
            doc.text(`Tanggal: ${new Date().toLocaleString('id-ID')}`, 105, 290, { align: 'center' });
            doc.text(`Halaman ${i} dari ${pageCount}`, 190, 290, { align: 'right' });
        }

        // ===== DOWNLOAD =====
        doc.save(`interview_report_${new Date().getTime()}.pdf`);
    };
}

// =========================================
// 🔹 Helper: Draw Section Header
// =========================================
function drawTableHeader(doc, title, yPos, xStart, blue) {
    doc.setFontSize(12);
    doc.setTextColor(...blue);
    doc.text(title, xStart, yPos);
    return yPos + 6;
}

// =========================================
// 🔹 Helper: Draw Table (Auto Page + Full Move)
// =========================================
function drawTable(doc, rows, headers, yPos, xStart, tableWidth, blue, cream) {
    const rowHeight = 7;
    const headerHeight = 8;
    const pageHeight = doc.internal.pageSize.getHeight();
    const bottomMargin = 20;
    const colCount = headers.length;
    const colWidth = tableWidth / colCount;

    // 🔹 Total tinggi tabel
    const tableHeight = headerHeight + rows.length * rowHeight;
    if (yPos + tableHeight > pageHeight - bottomMargin) {
        doc.addPage();
        yPos = 25;
    }

    // ===== HEADER =====
    doc.setFillColor(...blue);
    doc.setTextColor(255, 255, 255);
    doc.rect(xStart, yPos, tableWidth, headerHeight, 'F');
    doc.setFontSize(9);
    let xCursor = xStart + 5;
    headers.forEach((h, i) => {
        doc.text(h, xCursor, yPos + 5);
        xCursor += colWidth;
    });
    yPos += headerHeight;
    doc.setTextColor(0, 0, 0);

    // ===== ROWS =====
    rows.forEach((row, idx) => {
        // 🔹 Jika tabel baris berikutnya melewati batas halaman
        if (yPos + rowHeight > pageHeight - bottomMargin) {
            doc.addPage();
            yPos = 25;

            // Header ulang di halaman baru
            doc.setFillColor(...blue);
            doc.setTextColor(255, 255, 255);
            doc.rect(xStart, yPos, tableWidth, headerHeight, 'F');
            xCursor = xStart + 5;
            headers.forEach((h, i) => {
                doc.text(h, xCursor, yPos + 5);
                xCursor += colWidth;
            });
            yPos += headerHeight;
            doc.setTextColor(0, 0, 0);
        }

        // 🔹 Latar baris selang-seling
        if (idx % 2 === 0) {
            doc.setFillColor(...cream);
            doc.rect(xStart, yPos, tableWidth, rowHeight, 'F');
        }

        xCursor = xStart + 5;
        row.forEach((cell, i) => {
            const textLines = doc.splitTextToSize(String(cell), colWidth - 10);
            doc.text(textLines, xCursor, yPos + 5);
            xCursor += colWidth;
        });

        yPos += rowHeight;
    });

    return yPos;
}

// =========================================
// 🔹 Helper: Get Score Category
// =========================================
function getScoreCategoryText(score) {
    if (score >= 90) return 'Luar Biasa';
    if (score >= 80) return 'Bagus';
    if (score >= 70) return 'Cukup Bagus';
    return 'Perlu Peningkatan';
}

// =========================================================
// INITIALIZATION
// =========================================================
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Inisialisasi Dashboard...');
    loadJSONData();
});