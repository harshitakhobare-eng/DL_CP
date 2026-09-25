/**
 * MathTutor AI Frontend Application Logic.
 * Handles API integration, KaTeX rendering, Mermaid AST diagrams, and Chart.js.
 */

// Global State
let currentMistakeData = null;
let mistakesChart = null;

// Initialize when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  setupNavigation();
  setupSolveDropzone();
  setupVerifyDropzone();
  setupPracticeHandlers();
  mermaid.initialize({ startOnLoad: false, theme: "neutral" });
});

// Tab Navigation
function setupNavigation() {
  const tabs = document.querySelectorAll(".nav-item");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const targetTab = tab.getAttribute("data-tab");
      switchTab(targetTab);
    });
  });
}

function switchTab(tabName) {
  document.querySelectorAll(".nav-item").forEach(btn => {
    btn.classList.toggle("active", btn.getAttribute("data-tab") === tabName);
  });
  document.querySelectorAll(".tab-pane").forEach(pane => {
    pane.classList.toggle("active", pane.id === `tab-${tabName}`);
  });

  if (tabName === "history") loadHistory();
  if (tabName === "dashboard") loadDashboard();
}

// Helper: KaTeX Math Rendering
function renderMath(element) {
  if (window.renderMathInElement) {
    window.renderMathInElement(element, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "$", right: "$", display: false },
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false }
      ],
      throwOnError: false
    });
  }
}

// -------------------------------------------------------------
// TAB 1: SOLVE LOGIC
// -------------------------------------------------------------
function setupSolveDropzone() {
  const dropzone = document.getElementById("solve-dropzone");
  const fileInput = document.getElementById("solve-file-input");
  const solveBtn = document.getElementById("btn-solve-run");

  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", e => {
    e.preventDefault();
    dropzone.style.borderColor = "var(--accent)";
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.style.borderColor = "var(--rule)";
  });

  dropzone.addEventListener("drop", e => {
    e.preventDefault();
    dropzone.style.borderColor = "var(--rule)";
    if (e.dataTransfer.files.length) {
      handleSolveFileUpload(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", e => {
    if (e.target.files.length) {
      handleSolveFileUpload(e.target.files[0]);
    }
  });

  solveBtn.addEventListener("click", () => {
    const latex = document.getElementById("solve-latex-input").value.trim();
    if (latex) runSolve(latex);
  });
}

async function handleSolveFileUpload(file) {
  const formData = new FormData();
  formData.append("file", file);

  const dropzone = document.getElementById("solve-dropzone");
  dropzone.innerHTML = `<p>Recognizing handwritten expression with neural model...</p>`;

  try {
    const res = await fetch("/api/recognize", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok) throw new Error(data.detail || "Recognition failed.");

    // Update Preview
    if (data.preprocessed_image) {
      document.getElementById("solve-preview-container").style.display = "block";
      document.getElementById("solve-preview-img").src = `data:image/png;base64,${data.preprocessed_image}`;
    }

    // Set LaTeX
    document.getElementById("solve-latex-input").value = data.latex;

    // Confidence Banner
    const confBanner = document.getElementById("confidence-banner");
    confBanner.style.display = "block";
    const confPct = Math.round(data.confidence * 100);
    const badgeClass = confPct >= 80 ? "conf-high" : confPct >= 60 ? "conf-med" : "conf-low";

    let bannerHtml = `
      <div style="display:flex; align-items:center; gap:0.5rem; margin-top:0.4rem;">
        <span class="conf-badge ${badgeClass}">Recognition Confidence: ${confPct}%</span>
      </div>
    `;

    if (data.low_confidence_tokens && data.low_confidence_tokens.length > 0) {
      const warnings = data.low_confidence_tokens.map(t => t.warning).join(" • ");
      bannerHtml += `
        <div style="font-size:0.8rem; color:var(--warning); margin-top:0.3rem;">
          ⚠️ <strong>Visual Ambiguity:</strong> ${warnings}. Check the text box before solving.
        </div>
      `;
    }
    confBanner.innerHTML = bannerHtml;

    // Render AST if returned
    if (data.ast_mermaid) {
      renderMermaidAST(data.ast_mermaid);
    }

    // Run symbolic solve
    runSolve(data.latex);
  } catch (err) {
    alert("Error recognizing image: " + err.message);
  } finally {
    dropzone.innerHTML = `<p>Drag & drop handwritten image here, or <strong>browse file</strong></p>`;
  }
}

function loadSampleMath(latex) {
  document.getElementById("solve-latex-input").value = latex;
  document.getElementById("confidence-banner").style.display = "none";
  document.getElementById("solve-preview-container").style.display = "none";
  runSolve(latex);
}

async function runSolve(latex) {
  const resultArea = document.getElementById("solve-result-area");
  resultArea.innerHTML = `<p style="text-align:center; padding:1.5rem;">Solving equation deterministically...</p>`;

  try {
    const res = await fetch("/api/solve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ latex })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Solve failed.");

    // Update AST
    if (data.ast_mermaid) {
      renderMermaidAST(data.ast_mermaid);
    }

    // Render Solutions & Steps
    let html = `
      <div style="background:var(--ok-bg); border:1px solid #a7f3d0; border-radius:var(--r-sm); padding:0.75rem 1rem; margin-bottom:1rem;">
        <strong style="color:var(--ok);">Deterministic Symbolic Result:</strong>
        <div style="font-size:1.15rem; margin-top:0.3rem;">
          ${data.solutions_latex.map(s => `$${s}$`).join(" \\quad ")}
        </div>
      </div>
      <h3 style="font-size:0.95rem; margin-bottom:0.75rem;">Step-by-Step Mathematical Derivation:</h3>
    `;

    if (data.steps && data.steps.length > 0) {
      data.steps.forEach(s => {
        html += `
          <div class="step-item">
            <div class="step-header">
              <span>Step ${s.step_number}: ${s.operation}</span>
            </div>
            <div class="step-math">$${s.expression_after}$</div>
            <div class="step-expl">${s.explanation}</div>
          </div>
        `;
      });
    } else {
      html += `<p style="color:var(--ink-3);">Single-step transformation.</p>`;
    }

    resultArea.innerHTML = html;
    renderMath(resultArea);
  } catch (err) {
    resultArea.innerHTML = `<p style="color:var(--err); padding:1rem;">Failed to solve: ${err.message}</p>`;
  }
}

async function renderMermaidAST(mermaidCode) {
  const container = document.getElementById("mermaid-ast");
  try {
    const { svg } = await mermaid.render(`mermaid-svg-${Date.now()}`, mermaidCode);
    container.innerHTML = svg;
  } catch (err) {
    container.innerHTML = `<pre style="font-size:0.8rem;">${mermaidCode}</pre>`;
  }
}

// -------------------------------------------------------------
// TAB 2: CHECK MY SOLUTION LOGIC
// -------------------------------------------------------------
function setupVerifyDropzone() {
  const dropzone = document.getElementById("verify-dropzone");
  const fileInput = document.getElementById("verify-file-input");
  const verifyBtn = document.getElementById("btn-verify-run");

  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", e => {
    e.preventDefault();
    dropzone.style.borderColor = "var(--accent)";
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.style.borderColor = "var(--rule)";
  });

  dropzone.addEventListener("drop", e => {
    e.preventDefault();
    dropzone.style.borderColor = "var(--rule)";
    if (e.dataTransfer.files.length) {
      handleVerifyImage(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", e => {
    if (e.target.files.length) {
      handleVerifyImage(e.target.files[0]);
    }
  });

  verifyBtn.addEventListener("click", runTextVerify);
}

function loadSampleSolution(type) {
  const textarea = document.getElementById("verify-text-steps");
  if (type === "valid") {
    textarea.value = "2x + 5 = 15\n2x = 10\nx = 5";
  } else if (type === "dist") {
    textarea.value = "3(x + 4) = 21\n3x + 4 = 21\n3x = 17";
  } else if (type === "arith") {
    textarea.value = "2x + 5 = 15\n2x = 11\nx = 5.5";
  } else if (type === "trans") {
    textarea.value = "2x = 10\nx = 8";
  }
  runTextVerify();
}

async function handleVerifyImage(file) {
  const formData = new FormData();
  formData.append("file", file);

  const container = document.getElementById("verify-results-container");
  container.innerHTML = `<p style="text-align:center; padding:2rem;">Segmenting lines and recognizing handwritten steps...</p>`;

  try {
    const res = await fetch("/api/verify", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Verification failed.");
    displayVerificationResult(data);
  } catch (err) {
    container.innerHTML = `<p style="color:var(--err); padding:1rem;">Verification error: ${err.message}</p>`;
  }
}

async function runTextVerify() {
  const text = document.getElementById("verify-text-steps").value.trim();
  if (!text) return;

  const rawLines = text.split("\n").map(l => l.replace(/^Step\s*\d+\s*:\s*/i, "").trim()).filter(l => l.length > 0);
  if (rawLines.length === 0) return;

  const formData = new FormData();
  formData.append("steps_json", JSON.stringify(rawLines));

  const container = document.getElementById("verify-results-container");
  container.innerHTML = `<p style="text-align:center; padding:2rem;">Verifying mathematical deductions...</p>`;

  try {
    const res = await fetch("/api/verify", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Verification failed.");
    displayVerificationResult(data);
  } catch (err) {
    container.innerHTML = `<p style="color:var(--err); padding:1rem;">Verification error: ${err.message}</p>`;
  }
}

function displayVerificationResult(data) {
  const container = document.getElementById("verify-results-container");
  let html = "";

  if (data.is_correct) {
    html += `
      <div style="background:var(--ok-bg); border:1px solid #a7f3d0; border-radius:var(--r-md); padding:1rem; margin-bottom:1.25rem;">
        <h3 style="color:var(--ok); font-size:1.05rem;">🎉 Solution Verified Correct!</h3>
        <p style="font-size:0.85rem; color:var(--ink-2); margin-top:0.25rem;">All consecutive algebraic transformations are mathematically sound.</p>
      </div>
    `;
  } else {
    const errType = data.error_classification ? data.error_classification.error_type : "Mathematical Flaw";
    html += `
      <div style="background:var(--err-bg); border:1px solid #fecaca; border-radius:var(--r-md); padding:1rem; margin-bottom:1.25rem;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <h3 style="color:var(--err); font-size:1.05rem;">⚠️ Mathematical Error Detected at Step ${data.first_incorrect_step}</h3>
          <span style="background:var(--err); color:#fff; font-size:0.75rem; font-weight:700; padding:0.25rem 0.6rem; border-radius:9999px;">${errType}</span>
        </div>
        <p style="font-size:0.85rem; color:var(--ink-2); margin-top:0.4rem;">
          ${data.error_classification ? data.error_classification.explanation : "The step is not mathematically equivalent."}
        </p>
        <button id="btn-goto-explain" class="btn btn-primary" style="margin-top:0.75rem; font-size:0.8rem; padding:0.4rem 0.8rem;">
          📖 Explain My Mistake in Detail
        </button>
      </div>
    `;
  }

  // Render Steps
  html += `<h4 style="font-size:0.9rem; margin-bottom:0.75rem;">Step Audit:</h4>`;
  data.step_statuses.forEach(s => {
    const isError = !s.is_valid;
    const badgeColor = isError ? "var(--err)" : "var(--ok)";
    const badgeBg = isError ? "var(--err-bg)" : "var(--ok-bg)";
    const badgeText = isError ? "❌ Error" : "✓ Valid";

    html += `
      <div class="step-item ${isError ? 'step-error' : ''}">
        <div class="step-header">
          <span>Step ${s.step_number}</span>
          <span style="font-size:0.8rem; font-weight:600; padding:0.2rem 0.5rem; border-radius:4px; background:${badgeBg}; color:${badgeColor};">${badgeText}</span>
        </div>
        <div class="step-math">$${s.latex}$</div>
        <div class="step-expl">${s.status_message}</div>
      </div>
    `;
  });

  container.innerHTML = html;
  renderMath(container);

  // Setup "Explain My Mistake" button action
  if (!data.is_correct && data.explanation) {
    currentMistakeData = data.explanation;
    const explainBtn = document.getElementById("btn-goto-explain");
    if (explainBtn) {
      explainBtn.addEventListener("click", () => {
        populateExplainTab(data.explanation);
        switchTab("explain");
      });
    }
  }
}

// -------------------------------------------------------------
// TAB 3: EXPLAIN MY MISTAKE LOGIC
// -------------------------------------------------------------
function populateExplainTab(expl) {
  const container = document.getElementById("explain-content-area");
  const html = `
    <div class="error-breakdown-card">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem; border-bottom:1px solid #fecaca; padding-bottom:0.75rem;">
        <div>
          <span class="error-label">Mistake Taxonomy:</span>
          <h3 style="font-size:1.25rem; color:var(--err);">${expl.error_type}</h3>
        </div>
        <span style="font-size:0.85rem; font-weight:600; color:var(--ink-3);">Step ${expl.step_number}</span>
      </div>

      <div class="grid-2">
        <div class="error-field">
          <div class="error-label">1. What You Wrote:</div>
          <div class="error-val" style="color:var(--err); font-size:1.15rem;">$${expl["1_what_student_wrote"]}$</div>
        </div>

        <div class="error-field">
          <div class="error-label">2. What Was Expected:</div>
          <div class="error-val" style="color:var(--ok); font-size:1.15rem;">$${expl["2_what_was_expected"]}$</div>
        </div>
      </div>

      <div class="error-field" style="margin-top:0.75rem;">
        <div class="error-label">3. Why This Step Is Incorrect:</div>
        <div class="error-val">${expl["3_why_incorrect"]}</div>
      </div>

      <div class="error-field">
        <div class="error-label">4. Mathematical Rule Involved:</div>
        <div class="error-val"><strong>${expl["4_mathematical_rule"]}</strong></div>
      </div>

      <div class="error-field">
        <div class="error-label">5. Corrected Transformation:</div>
        <div class="error-val" style="background:#fff; padding:0.5rem 0.75rem; border-radius:6px; border:1px solid #a7f3d0; color:var(--ok); font-size:1.1rem;">
          $${expl["5_corrected_step"]}$
        </div>
      </div>

      <div class="error-field" style="margin-top:0.75rem;">
        <div class="error-label">6. Tip to Avoid This Mistake:</div>
        <div class="error-val" style="color:var(--gray-800);">💡 ${expl["6_avoidance_tip"]}</div>
      </div>
    </div>
  `;

  container.innerHTML = html;
  renderMath(container);
}

// -------------------------------------------------------------
// TAB 4: MISTAKE HISTORY LOGIC
// -------------------------------------------------------------
async function loadHistory() {
  const tbody = document.getElementById("history-table-body");
  try {
    const res = await fetch("/api/mistakes");
    const data = await res.json();

    if (!data || data.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" class="table-empty">No mistakes recorded yet. All your attempts are saved here.</td></tr>`;
      return;
    }

    tbody.innerHTML = data.map(m => `
      <tr>
        <td style="font-size:0.8rem; color:var(--ink-3);">${new Date(m.timestamp).toLocaleDateString()} ${new Date(m.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</td>
        <td><strong>${m.topic}</strong></td>
        <td>$${m.problem}$</td>
        <td style="color:var(--err);">$${m.student_step || '—'}$</td>
        <td style="color:var(--ok);">$${m.correct_step || '—'}$</td>
        <td><span style="font-size:0.75rem; padding:0.2rem 0.5rem; background:var(--err-bg); color:var(--err); border-radius:4px; font-weight:600;">${m.mistake_type}</span></td>
        <td>
          <button class="btn btn-ghost" style="font-size:0.75rem; padding:0.3rem 0.6rem;" onclick='viewHistoricMistake(${JSON.stringify(m)})'>Explain</button>
        </td>
      </tr>
    `).join("");

    renderMath(tbody);
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="7" style="color:var(--err); text-align:center;">Failed to load history: ${err.message}</td></tr>`;
  }
}

function viewHistoricMistake(m) {
  const expl = {
    error_type: m.mistake_type,
    step_number: m.step_number,
    "1_what_student_wrote": m.student_step,
    "2_what_was_expected": m.correct_step,
    "3_why_incorrect": m.explanation,
    "4_mathematical_rule": "Standard Mathematical Law",
    "5_corrected_step": m.correct_step,
    "6_avoidance_tip": "Double-check this transformation step carefully.",
  };
  populateExplainTab(expl);
  switchTab("explain");
}

// -------------------------------------------------------------
// TAB 5: PERSONALIZED PRACTICE LOGIC
// -------------------------------------------------------------
function setupPracticeHandlers() {
  const genBtn = document.getElementById("btn-generate-practice");
  genBtn.addEventListener("click", generatePractice);
}

async function generatePractice() {
  const cat = document.getElementById("practice-category-select").value;
  const diff = document.getElementById("practice-difficulty-select").value;
  const container = document.getElementById("practice-questions-container");

  container.innerHTML = `<p style="text-align:center; padding:2rem;">Synthesizing targeted mathematical problems...</p>`;

  try {
    const res = await fetch("/api/practice/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mistake_type: cat || null, difficulty: diff, num_questions: 3 })
    });
    const data = await res.json();

    document.getElementById("practice-subtitle").innerText =
      `Targeted Weakness: ${data.targeted_weakness} (${data.difficulty} difficulty)`;

    let html = "";
    data.problems.forEach((p, idx) => {
      html += `
        <div class="practice-item" id="pbox-${p.problem_id}">
          <div style="display:flex; justify-content:space-between; font-size:0.85rem; color:var(--ink-3);">
            <span>Question ${idx + 1} • ${p.topic}</span>
            <span style="font-weight:600;">Weakness: ${p.targeted_mistake_type}</span>
          </div>

          <div class="practice-math">$${p.problem_latex}$</div>

          <div style="display:flex; gap:0.75rem; max-width:480px; margin:0 auto;">
            <input type="text" class="text-input" id="ans-${p.problem_id}" placeholder="Enter answer (e.g. x = 5)" style="margin-bottom:0;" />
            <button class="btn btn-primary" onclick="checkPracticeAnswer('${p.problem_id}', '${encodeURIComponent(JSON.stringify(p))}')">Submit</button>
          </div>

          <div id="feedback-${p.problem_id}" style="margin-top:0.75rem; text-align:center;"></div>

          <div style="text-align:center; margin-top:0.5rem;">
            <button class="btn btn-ghost" style="font-size:0.75rem; padding:0.25rem 0.6rem;" onclick="togglePracticeSteps('${p.problem_id}')">
              Show Step-by-Step Solution
            </button>
          </div>

          <div id="steps-${p.problem_id}" style="display:none; margin-top:0.75rem; text-align:left;">
            <h4 style="font-size:0.85rem; margin-bottom:0.5rem;">Verified Derivation:</h4>
            ${p.steps.map(s => `
              <div class="step-item">
                <div class="step-header"><span>Step ${s.step_number}: ${s.operation}</span></div>
                <div class="step-math">$${s.expression_after}$</div>
                <div class="step-expl">${s.explanation}</div>
              </div>
            `).join("")}
          </div>
        </div>
      `;
    });

    container.innerHTML = html;
    renderMath(container);
  } catch (err) {
    container.innerHTML = `<p style="color:var(--err); padding:1rem;">Failed to generate practice: ${err.message}</p>`;
  }
}

async function checkPracticeAnswer(qid, encodedProblem) {
  const p = JSON.parse(decodeURIComponent(encodedProblem));
  const studentAns = document.getElementById(`ans-${qid}`).value.trim();
  const feedback = document.getElementById(`feedback-${qid}`);

  if (!studentAns) return;

  // Normalize answers
  const normStudent = studentAns.replace(/\s+/g, "").toLowerCase();
  const isMatch = p.solutions_latex.some(sol => {
    const normSol = sol.replace(/[{}\\s]/g, "").toLowerCase();
    return normStudent.includes(normSol) || normSol.includes(normStudent);
  }) || p.final_answer.replace(/\s+/g, "").toLowerCase().includes(normStudent);

  if (isMatch) {
    feedback.innerHTML = `<span style="color:var(--ok); font-weight:600;">✓ Correct! Excellent work!</span>`;
  } else {
    feedback.innerHTML = `
      <span style="color:var(--err); font-weight:600;">Incorrect.</span>
      <span style="font-size:0.85rem; color:var(--ink-3); margin-left:0.5rem;">Correct: $${p.solutions_latex[0]}$</span>
    `;
    renderMath(feedback);
  }

  // Record practice result
  await fetch("/api/practice/submit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      topic: p.topic,
      problem_latex: p.problem_latex,
      solution_latex: p.solutions_latex[0],
      student_answer: studentAns,
      is_correct: isMatch,
      mistake_type: p.targeted_mistake_type
    })
  });
}

function togglePracticeSteps(qid) {
  const el = document.getElementById(`steps-${qid}`);
  el.style.display = el.style.display === "none" ? "block" : "none";
}

// -------------------------------------------------------------
// TAB 6: PROGRESS DASHBOARD LOGIC
// -------------------------------------------------------------
async function loadDashboard() {
  try {
    const res = await fetch("/api/dashboard");
    const data = await res.json();

    document.getElementById("dash-total-attempts").innerText = data.total_attempts;
    document.getElementById("dash-accuracy").innerText = `${data.overall_accuracy}%`;
    document.getElementById("dash-recent-accuracy").innerText = `${data.recent_accuracy}%`;
    document.getElementById("dash-streak").innerText = `${data.streak_days} Day${data.streak_days === 1 ? '' : 's'} 🔥`;

    if (data.recommendation) {
      document.getElementById("dash-recommendation-text").innerText = data.recommendation;
    }

    // Render Mistakes Chart
    renderMistakesChart(data.mistakes_by_category);

    // Render Topic Mastery
    const topicList = document.getElementById("topic-mastery-list");
    if (data.topic_performance && data.topic_performance.length > 0) {
      topicList.innerHTML = data.topic_performance.map(t => `
        <div style="margin-bottom:1rem;">
          <div style="display:flex; justify-content:space-between; font-size:0.85rem; margin-bottom:0.25rem;">
            <span><strong>${t.topic}</strong> (${t.total_correct}/${t.total_attempts})</span>
            <span style="font-weight:600;">${t.accuracy}%</span>
          </div>
          <div style="background:var(--gray-200); height:8px; border-radius:9999px; overflow:hidden;">
            <div style="background:var(--accent); height:100%; width:${t.accuracy}%;"></div>
          </div>
        </div>
      `).join("");
    } else {
      topicList.innerHTML = `<p style="color:var(--ink-3); text-align:center;">No topic data recorded yet.</p>`;
    }
  } catch (err) {
    console.error("Error loading dashboard:", err);
  }
}

function renderMistakesChart(categories) {
  const ctx = document.getElementById("chart-mistakes").getContext("2d");
  const labels = Object.keys(categories || {});
  const counts = Object.values(categories || {});

  if (mistakesChart) mistakesChart.destroy();

  if (labels.length === 0) {
    labels.push("No Mistakes Recorded");
    counts.push(0);
  }

  mistakesChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: "Mistake Count",
        data: counts,
        backgroundColor: "rgba(239, 68, 68, 0.7)",
        borderColor: "rgba(239, 68, 68, 1)",
        borderWidth: 1,
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      indexAxis: "y",
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: {
          beginAtZero: true,
          ticks: { stepSize: 1 }
        }
      }
    }
  });
}
