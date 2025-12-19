// src/App.js
import React, { useState } from "react";
import "./App.css";

const API_BASE_URL = "http://localhost:8008";

// 제품별 신규 결함(제외 결함) 후보
const NEW_DEFECT_CANDIDATES = {
  capsule: ["crack", "squeeze"],
  tile: ["oil", "glue_strip"],
  leather: ["cut", "poke"],
};

// 제품별 "기존 결함과 유사함" 선택 시 보여줄 결함 유형들
// ▶ 여기에는 제외 결함은 넣지 않는다.
const EXISTING_DEFECT_TYPES = {
  capsule: ["scratch", "contamination", "misprint"],
  tile: ["crack", "scratch"],
  leather: ["color", "fold", "scratch"],
};

function App() {
  const [productType, setProductType] = useState("capsule");

  const [messages, setMessages] = useState([
    {
      id: 1,
      role: "assistant",
      text: "안녕하세요! Factory Defect Agent입니다. 가운데 패널에서 이미지를 업로드한 뒤 분석하거나 채팅창에 질문을 입력해 보세요.",
    },
  ]);

  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);

  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedPreviewUrl, setSelectedPreviewUrl] = useState(null);
  const [selectedFileName, setSelectedFileName] = useState("");

  // 1단계 / 2단계 / 3단계 결과
  const [stage1Result, setStage1Result] = useState(null);
  const [stage2Result, setStage2Result] = useState(null);
  const [stage3Result, setStage3Result] = useState(null);

  // 3단계에서 사용자 선택 상태
  const [adaptStep, setAdaptStep] = useState(null); // "similar" | "new" | null
  const [selectedSimilarDefect, setSelectedSimilarDefect] = useState("");
  const [selectedNewDefectName, setSelectedNewDefectName] = useState("");

  // ---------------- 공통: 메시지 추가 ----------------
  const pushMessage = (role, text) => {
    setMessages((prev) => [...prev, { id: prev.length + 1, role, text }]);
  };

  // ---------------- 이미지 업로드 ----------------
  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const url = URL.createObjectURL(file);

    setSelectedFile(file);
    setSelectedFileName(file.name);
    setSelectedPreviewUrl(url);

    // 새 이미지 → 파이프라인/선택/결과/채팅 초기화
    setStage1Result(null);
    setStage2Result(null);
    setStage3Result(null);
    setAdaptStep(null);
    setSelectedSimilarDefect("");
    setSelectedNewDefectName("");

    setMessages([
      {
        id: 1,
        role: "assistant",
        text: `새로운 이미지 "${file.name}" 이(가) 업로드되었습니다. 분류를 실행해 보세요.`,
      },
    ]);
  };

  // ---------------- 1단계: 분류 API 호출 ----------------
  const runStage1 = async () => {
    if (!selectedFile) {
      pushMessage(
        "assistant",
        "먼저 분석할 이미지를 가운데 패널에서 업로드해 주세요."
      );
      return;
    }

    // 1단계 다시 실행 → 이후 단계 reset
    setStage2Result(null);
    setStage3Result(null);
    setAdaptStep(null);
    setSelectedSimilarDefect("");
    setSelectedNewDefectName("");

    setIsSending(true);
    pushMessage(
      "assistant",
      "정상/결함 및 클래스 분류를 수행 중입니다…"
    );

    try {
      const formData = new FormData();
      formData.append("category", productType);
      formData.append("file", selectedFile);

      const res = await fetch(`${API_BASE_URL}/api/classify`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `status ${res.status}`);
      }

      const data = await res.json();
      setStage1Result(data);

      pushMessage(
        "assistant",
        `분류 결과: "${data.filename}" 은(는) [${data.category}] 제품이며 예측된 불량 유형은 "${data.predicted_defect}" 입니다.`
      );
    } catch (err) {
      console.error("Stage1 error:", err);
      pushMessage(
        "assistant",
        "⚠️ 분류 중 오류가 발생했습니다. 백엔드 로그를 확인해 주세요."
      );
    } finally {
      setIsSending(false);
    }
  };

  // ---------------- 2단계: SegFormer 위치 탐지 ----------------
  const runStage2 = async () => {
    if (!selectedFile) {
      pushMessage(
        "assistant",
        "먼저 분석할 이미지를 가운데 패널에서 업로드해 주세요."
      );
      return;
    }
    if (!stage1Result) {
      pushMessage("assistant", "먼저 분류를 실행해 주세요.");
      return;
    }

    // 2단계 다시 실행 → 3단계 reset
    setStage3Result(null);
    setAdaptStep(null);
    setSelectedSimilarDefect("");
    setSelectedNewDefectName("");

    setIsSending(true);
    pushMessage("assistant", "결함 위치 마스크를 생성 중입니다…");

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("product_type", productType);
      formData.append("pred_class", stage1Result.predicted_defect || "");

      const res = await fetch(`${API_BASE_URL}/api/segment`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `status ${res.status}`);
      }

      const data = await res.json();
      setStage2Result(data);

      pushMessage(
        "assistant",
        "결함 위치 마스크 생성이 완료되었습니다."
      );

      // 🔹 2단계 LLM 설명이 있으면 채팅에 추가
      if (data.description) {
        pushMessage("assistant", data.description);
      }
    } catch (err) {
      console.error("Stage2 error:", err);
      pushMessage(
        "assistant",
        "⚠️ 위치 탐지 중 오류가 발생했습니다. 백엔드 로그를 확인해 주세요."
      );
    } finally {
      setIsSending(false);
    }
  };

  // ---------------- 3단계: 적응학습 분석 ----------------
  const runStage3 = async () => {
    if (!selectedFile) {
      pushMessage(
        "assistant",
        "먼저 분석할 이미지를 가운데 패널에서 업로드해 주세요."
      );
      return;
    }

    setIsSending(true);
    pushMessage(
      "assistant",
      "적응학습 분석을 수행 중입니다…"
    );

    try {
      const formData = new FormData();
      formData.append("category", productType);
      formData.append("file", selectedFile);

      const res = await fetch(`${API_BASE_URL}/api/adapt`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `status ${res.status}`);
      }

      const data = await res.json();
      setStage3Result(data);
      setAdaptStep(null);
      setSelectedSimilarDefect("");
      setSelectedNewDefectName("");

      const pGood = data.p_good ?? 0;
      let desc = "";
      if (pGood >= 0.8) {
        desc = "모델은 이 샘플을 거의 정상으로 판단하고 있습니다.";
      } else if (pGood >= 0.6) {
        desc = "대체로 정상으로 보이지만 약간의 불확실성이 있습니다.";
      } else if (pGood >= 0.4) {
        desc =
          "정상/불량이 애매한 샘플입니다. 새로운 결함 유형일 가능성을 고려할 수 있습니다.";
      } else if (pGood >= 0.2) {
        desc = "불량일 가능성이 높은 샘플로 보입니다.";
      } else {
        desc =
          "모델은 이 샘플을 거의 확실하게 '불량'으로 판단하고 있습니다. 기존 학습에 포함되지 않은 새로운 결함 유형일 가능성이 커 보입니다.";
      }

      // 3단계 분석 결과 설명
      pushMessage("assistant", `분석 결과: ${desc}`);

      // 사용자 안내 메시지
      pushMessage(
        "assistant",
        "이제 가운데 패널의 적응학습 영역에서 '기존 결함과 유사함' 또는 '신규 결함으로 추가' 버튼을 선택해 주세요. " +
          '버튼 아래에 보이는 후보 이름을 클릭해도 되고, 채팅창에 직접 "이 결함을 XXX 유형으로 등록해줘" 와 같이 입력해도 됩니다.'
      );
    } catch (err) {
      console.error("Stage3 error:", err);
      pushMessage(
        "assistant",
        "⚠️ 적응학습 분석 중 오류가 발생했습니다. 백엔드 로그를 확인해 주세요."
      );
    } finally {
      setIsSending(false);
    }
  };

  // ---------------- 3단계: "기존 결함과 유사함" 선택 핸들러 ----------------
  const handleSimilarDefectChoice = async (defectName) => {
    setSelectedSimilarDefect(defectName);

    const msg = `이 결함 이미지를 기존 결함 유형 "${defectName}"과(와) 가장 유사한 사례로 표시해줘. 여기서 "${defectName}"은 제조 공정에서 사용하는 결함 클래스 이름이야.`;
    pushMessage("user", msg);
    setIsSending(true);

    try {
      const res = await fetch(`${API_BASE_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg }),
      });
      const data = await res.json();
      pushMessage(
        "assistant",
        data.reply || "기존 결함 유형과의 매핑 결과를 가져오지 못했습니다."
      );
    } catch (e) {
      console.error(e);
      pushMessage(
        "assistant",
        "⚠️ 기존 결함과 유사한 유형으로 표시하는 중 오류가 발생했습니다."
      );
    } finally {
      setIsSending(false);
    }
  };

  // ---------------- 3단계: "신규 결함으로 추가" 선택 핸들러 ----------------
  const handleNewDefectChoice = async (defectName) => {
    setSelectedNewDefectName(defectName);

    const msg = `이 결함 이미지를 새로운 결함 유형 "${defectName}"으로 등록해줘.`;
    pushMessage("user", msg);
    setIsSending(true);

    try {
      const res = await fetch(`${API_BASE_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg }),
      });
      const data = await res.json();
      pushMessage(
        "assistant",
        data.reply || "신규 결함 등록에 대한 LLM 응답을 가져오지 못했습니다."
      );
    } catch (e) {
      console.error(e);
      pushMessage(
        "assistant",
        "⚠️ 신규 결함 등록 요청 중 오류가 발생했습니다."
      );
    } finally {
      setIsSending(false);
    }
  };

  // ---------------- LLM Chat + LangGraph Agent Chat ----------------
  const handleSend = async (e) => {
    e.preventDefault();

    const trimmed = input.trim();
    if (!trimmed) return;

    // 사용자 메시지 채팅창에 추가
    pushMessage("user", trimmed);
    setInput("");
    setIsSending(true);

    try {
      // -------------------------------------------------------
      // 1) 이미지가 없으면 → 기존 /api/chat 사용
      // -------------------------------------------------------
      if (!selectedFile) {
        const res = await fetch(`${API_BASE_URL}/api/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: trimmed }),
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const data = await res.json();
        pushMessage("assistant", data.reply || "(응답 없음)");
        return;
      }

      // -------------------------------------------------------
      // 2) 이미지가 있음 → LangGraph Agent (/api/agent-chat)
      // -------------------------------------------------------
      const formData = new FormData();
      formData.append("message", trimmed);
      formData.append("category", productType);
      formData.append("file", selectedFile);

      const res = await fetch(`${API_BASE_URL}/api/agent-chat`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `status ${res.status}`);
      }

      const data = await res.json();

      // LangGraph 에이전트의 자연어 답변
      pushMessage("assistant", data.reply || "(응답 없음)");

      // 🔥 LangGraph 결과를 자동으로 UI 1·2·3단계에 반영
      if (data.cls_result) setStage1Result(data.cls_result);
      if (data.seg_result) setStage2Result(data.seg_result);
      if (data.adapt_result) setStage3Result(data.adapt_result);

    } catch (err) {
      console.error("Agent Chat Error:", err);
      pushMessage(
        "assistant",
        "⚠️ 에이전트 대화 처리 중 오류가 발생했습니다. 백엔드 로그를 확인해 주세요."
      );
    } finally {
      setIsSending(false);
    }
  };

// ---------------- 통합 실행: LangGraph 파이프라인 ----------------
  const runPipeline = async () => {
    if (!selectedFile) {
      pushMessage(
        "assistant",
        "먼저 분석할 이미지를 가운데 패널에서 업로드해 주세요."
      );
      return;
    }

    setIsSending(true);
    pushMessage("assistant", "LangGraph 기반 통합 파이프라인을 실행합니다…");

    try {
      const formData = new FormData();
      formData.append("category", productType);
      formData.append("file", selectedFile);

      const res = await fetch(`${API_BASE_URL}/api/pipeline`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `status ${res.status}`);
      }

      const data = await res.json();

      // 백엔드 LangGraph 응답 매핑
      setStage1Result(data.cls_result || null);
      setStage2Result(data.seg_result || null);
      setStage3Result(data.adapt_result || null);

      pushMessage(
        "assistant",
        "통합 파이프라인 실행이 완료되었습니다. 좌측 패널에서 1·2·3단계 결과를 확인하세요!"
      );
    } catch (err) {
      console.error("Pipeline error:", err);
      pushMessage(
        "assistant",
        "⚠️ 통합 파이프라인 실행 중 오류가 발생했습니다. 백엔드 로그를 확인해 주세요."
      );
    } finally {
      setIsSending(false);
    }
  };
  // ---------------- 보고서 저장 & PDF ----------------
  const handleSaveReport = async () => {
    if (!stage1Result) {
      alert("먼저 분석을 수행해 주세요.");
      return;
    }

    const payload = {
      product_type: productType,
      file_name: stage1Result.filename,
      predicted_defect: stage1Result.predicted_defect,
      stage1_summary: `제품: ${stage1Result.category}, 예측 결함: ${stage1Result.predicted_defect}`,
      segmentation_summary: stage2Result?.description || "",
      adapt_summary: stage3Result?.summary || "",
      llm_description: "",
      orig_image_url: stage2Result?.overlay_url || null,
      mask_image_url: stage2Result?.mask_url || null,
    };

    try {
      await fetch(`${API_BASE_URL}/api/defects/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      window.open(`${API_BASE_URL}/api/defects/report`, "_blank");
    } catch (e) {
      console.error(e);
      alert("보고서 생성 중 오류가 발생했습니다. 백엔드 로그를 확인해 주세요.");
    }
  };

  // 기존 결함 목록에서 신규(제외) 결함 후보는 제거
  const existingDefectList = (
    EXISTING_DEFECT_TYPES[productType] || []
  ).filter(
    (name) => !(NEW_DEFECT_CANDIDATES[productType] || []).includes(name)
  );

  return (
    <div className="fh-app">
      <div className="fh-shell">
        {/* ===== 왼쪽 사이드바 ===== */}
        <aside className="fh-sidebar">
          <div className="fh-logo-block">
            <div className="fh-logo-dot" />
            <div>
              <div className="fh-logo-text">Factory Defect Agent</div>
              <div className="fh-logo-sub">Capsule/Tile/Leather</div>
            </div>
          </div>

          <div className="fh-section">
            <div className="fh-section-title">PRODUCT</div>
            <select
              className="fh-select"
              value={productType}
              onChange={(e) => {
                const next = e.target.value;
                setProductType(next);

                // 제품 바뀌면 전체 상태/채팅 초기화
                setSelectedFile(null);
                setSelectedFileName("");
                setSelectedPreviewUrl(null);
                setStage1Result(null);
                setStage2Result(null);
                setStage3Result(null);
                setAdaptStep(null);
                setSelectedSimilarDefect("");
                setSelectedNewDefectName("");

                setMessages([
                  {
                    id: 1,
                    role: "assistant",
                    text: `제품을 ${next}로 변경했습니다. 가운데 패널에서 새 이미지를 업로드한 뒤 다시 실행해 주세요.`,
                  },
                ]);
              }}
            >
              <option value="capsule">Capsule</option>
              <option value="tile">Tile</option>
              <option value="leather">Leather</option>
            </select>
          </div>

          <div className="fh-section">
            <div className="fh-section-title">PIPELINE STEPS</div>
            <ol className="fh-step-list">
              <li>정상/불량 및 클래스 분류</li>
              <li>결함 위치 탐지 및 설명</li>
              <li>신규 결함 적응 학습</li>
              <li>LLM 기반 대화형 에이전트</li>
            </ol>
          </div>

          {/* 🔥 왼쪽 빈 공간에 보고서 버튼 */}
          <div className="fh-section">
            <div className="fh-section-title">DEFECT REPORT</div>
            <p style={{ fontSize: 12, color: "#6b7280", marginBottom: 8 }}>
              현재 분석 결과를 기반으로 PDF 형식의 결함 보고서를 생성합니다.
            </p>
            <button
              type="button"
              className="fh-report-btn"
              onClick={handleSaveReport}
              disabled={!stage1Result}
            >
              📋 PDF 보고서 다운로드
            </button>
          </div>

          <div className="fh-section fh-section-note">
            <div className="fh-note-title">💡 TIP</div>
            <p>
              이미지를 업로드 한 뒤 분류 실행 / 적응학습 버튼을 눌러 결함을 분석해보세요.<br />
              결함을 분석한 뒤에는 결과에 대한 보고서를 출력할 수 있습니다.<br />
              한 번에 통합 분석도 가능합니다.<br />
              지금 바로 실행해보세요!<br />
            </p>
          </div>
        </aside>

        {/* ===== 가운데: 이미지 & 분석 패널 ===== */}
        <aside className="fh-detail-pane">
          <header className="fh-detail-header">
            <div className="fh-detail-title">📊 결함 분석</div>
            <div className="fh-detail-subtitle">
              결함 이미지 분석을 할 수 있는 영역입니다.
            </div>
          </header>

          <section className="fh-detail-body">
            {/* 업로드 폼 */}
            <div className="fh-form-group">
              <label>분석할 이미지 업로드</label>
              <label className="fh-file-input">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                />
                <span>{selectedFileName || "클릭하여 이미지 선택"}</span>
              </label>
            </div>

            {/* 미리보기 */}
            {selectedPreviewUrl && (
              <div className="fh-image-preview-block">
                <div className="fh-preview-label">입력 이미지 미리보기</div>
                <img
                  src={selectedPreviewUrl}
                  alt="selected"
                  className="fh-image-preview"
                />
              </div>
            )}

            {/* 파이프라인 버튼 */}
            <div className="fh-form-group">
              <label>파이프라인 실행</label>
              <div className="fh-button-row">
                <button onClick={runStage1} disabled={isSending}>
                  분류 실행
                </button>
                <button
                  onClick={runStage2}
                  disabled={isSending || !stage1Result || !selectedFile}
                >
                  결함 위치 탐지
                </button>
                <button
                  onClick={runStage3}
                  disabled={isSending || !selectedFile || !!stage1Result}
                >
                  적응학습
                </button>
                <button
                  onClick={runPipeline}
                  disabled={isSending || !selectedFile}
                >
                  통합 실행
                </button>
              </div>
            </div>

            {/* 1단계 결과 */}
            <div className="fh-analysis-section">
              <h4>분류 결과</h4>
              {stage1Result ? (
                <div className="fh-analysis-card">
                  <div>
                    📦 Product : <b>{stage1Result.category}</b>
                  </div>
                  <div>
                    🖇️ File : <b>{stage1Result.filename}</b>
                  </div>
                  <div>
                    🔍 Predicted defect :{" "}
                    <b>{stage1Result.predicted_defect}</b>
                  </div>
                </div>
              ) : (
                <div className="fh-analysis-placeholder">
                  아직 실행되지 않았습니다. 이미지를 업로드한 뒤
                  <b> 분류 실행</b> 버튼을 눌러주세요.
                </div>
              )}
            </div>

            {/* 2단계 결과 */}
            <div className="fh-analysis-section">
              <h4>결함 위치 탐지 및 설명</h4>
              {stage2Result ? (
                <div className="fh-analysis-card">
                  <div className="fh-mask-placeholder">
                    {stage2Result.mask_url || stage2Result.overlay_url ? (
                      <img
                        src={`${API_BASE_URL}${
                          stage2Result.mask_url || stage2Result.overlay_url
                        }`}
                        alt="defect mask"
                        style={{ maxWidth: "100%" }}
                      />
                    ) : (
                      "마스크가 생성되었습니다."
                    )}
                  </div>
                </div>
              ) : (
                <div className="fh-analysis-placeholder">
                  분류 실행 후 <b>결함 위치 탐지</b> 버튼을 누르면 결함
                  마스크가 표시되고 결함에 대해 설명해 드립니다.
                </div>
              )}
            </div>

            {/* 3단계 결과 */}
            <div className="fh-analysis-section">
              <h4>적응학습 결과</h4>
              {stage3Result ? (
                <div className="fh-analysis-card">
                  <div>
                    모델 기준 정상 확률:{" "}
                    <b>{(stage3Result.p_good * 100).toFixed(1)}%</b>
                  </div>
                  <div>
                    모델 기준 불량 확률:{" "}
                    <b>{(stage3Result.p_defect * 100).toFixed(1)}%</b>
                  </div>

                  <div className="fh-button-row" style={{ marginTop: 8 }}>
                    <button
                      type="button"
                      onClick={() => setAdaptStep("similar")}
                    >
                      기존 결함과 유사함
                    </button>
                    <button
                      type="button"
                      onClick={() => setAdaptStep("new")}
                    >
                      신규 결함으로 추가
                    </button>
                  </div>

                  {/* 기존 결함과 유사함 */}
                  {adaptStep === "similar" && (
                    <>
                      <div style={{ marginTop: 8, fontSize: 12 }}>
                        기존 결함 유형 중에서 가장 가까운 유형을 선택해 주세요.
                        또는 채팅창에 직접{" "}
                        <b>"이 결함을 ○○ 유형과 유사하다고 표시해줘"</b> 라고
                        입력해도 됩니다.
                      </div>
                      <div className="fh-chip-row">
                        {existingDefectList.map((name) => (
                          <div
                            key={name}
                            className={`fh-chip ${
                              selectedSimilarDefect === name
                                ? "fh-chip-selected"
                                : ""
                            }`}
                            onClick={() => handleSimilarDefectChoice(name)}
                          >
                            {name}
                          </div>
                        ))}
                      </div>
                    </>
                  )}

                  {/* 신규 결함으로 추가 */}
                  {adaptStep === "new" && (
                    <>
                      <div style={{ marginTop: 8, fontSize: 12 }}>
                        이 결함을 새로운 유형으로 등록하고 싶다면 아래에서
                        사용할 이름을 선택하거나, 채팅창에{" "}
                        <b>"이 결함을 ~ 유형으로 등록해줘"</b> 라고 직접
                        입력해도 됩니다.
                      </div>
                      <div className="fh-chip-row">
                        {NEW_DEFECT_CANDIDATES[productType].map((name) => (
                          <div
                            key={name}
                            className={`fh-chip ${
                              selectedNewDefectName === name
                                ? "fh-chip-selected"
                                : ""
                            }`}
                            onClick={() => handleNewDefectChoice(name)}
                          >
                            {name}
                          </div>
                        ))}
                      </div>
                    </>
                  )}
                </div>
              ) : (
                <div className="fh-analysis-placeholder">
                  이미지 업로드 후{" "}
                  <b>적응학습</b> 버튼을 누르면 신규 결함을 파악하고 결함 유형을 추가할 수 있습니다. 
                </div>
              )}
            </div>
          </section>
        </aside>

        {/* ===== 오른쪽: 챗 영역 ===== */}
        <main className="fh-chat-pane">
          <header className="fh-chat-header">
            <div className="fh-chat-title">💻 AI 에이전트와의 대화 </div>
            <div className="fh-chat-subtitle">
              제품: <b>{productType}</b>
            </div>
          </header>

          <section className="fh-chat-messages">
            {messages.map((m) => (
              <div
                key={m.id}
                className={
                  m.role === "user"
                    ? "fh-msg-row fh-msg-user"
                    : "fh-msg-row fh-msg-assistant"
                }
              >
                <div className="fh-msg-avatar">
                  {m.role === "assistant" ? (
                    <img
                      src="/robot.png"
                      alt="agent"
                      className="fh-avatar-img"
                    />
                  ) : (
                    <span></span>
                  )}
                </div>
                <div className="fh-msg-body">
                  <div className="fh-msg-role">
                    {m.role === "user" ? "사용자" : "Factory Defect Agent"}
                  </div>
                  <div className="fh-msg-text">{m.text}</div>
                </div>
              </div>
            ))}
          </section>

          <footer className="fh-chat-input-bar">
            <form className="fh-chat-form" onSubmit={handleSend}>
              <input
                className="fh-chat-input"
                placeholder="무엇이든 물어보세요."
                value={input}
                onChange={(e) => setInput(e.target.value)}
              />
              <button className="fh-chat-send" disabled={isSending}>
                {isSending ? "생각 중…" : "Send"}
              </button>
            </form>
            <div className="fh-chat-hint">
              ❓ 궁금한 사항이 있으면 언제든지 물어보세요! 채팅으로도 대응이 가능합니다.
            </div>
          </footer>
        </main>
      </div>
    </div>
  );
}


export default App;


