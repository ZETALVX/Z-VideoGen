// ===== Helper API =====
async function postGenerate(payload){
  const res = await fetch("/generate", {method:"POST", body: payload});
  return res.json();
}

async function loadLoras(){
  try{
    const r = await fetch("/api/lora-list");
    const j = await r.json();
    const sel = document.getElementById("lora");
    if(!sel) return;
    sel.innerHTML = "";
    (j.loras || ["none"]).forEach(name=>{
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      sel.appendChild(opt);
    });
  }catch(e){
    console.warn("LoRA list failed:", e);
  }
}
window.addEventListener("DOMContentLoaded", loadLoras);

// ===== Generazione singola =====
const form = document.getElementById("gen-form");
if(form){
  form.addEventListener("submit", async (e)=>{
    e.preventDefault();
    const fd = new FormData();
    fd.append("mode", document.getElementById("mode").value);
    fd.append("prompt", document.getElementById("prompt").value);
    fd.append("negative_prompt", document.getElementById("neg").value);
    fd.append("width", document.getElementById("width").value);
    fd.append("height", document.getElementById("height").value);
    fd.append("frames", document.getElementById("frames").value);
    fd.append("steps", document.getElementById("steps").value);
    fd.append("cfg", document.getElementById("cfg").value);
    fd.append("seed", document.getElementById("seed").value);
    fd.append("strength", document.getElementById("strength").value);
    fd.append("motion_bucket_id", document.getElementById("motion_bucket_id").value);
    fd.append("lora", document.getElementById("lora").value);
    fd.append("lora_scale", document.getElementById("lora_scale").value);
    fd.append("sched", document.getElementById("sched").value);
    fd.append("offload", document.getElementById("offload").checked ? "1":"0");

    const img = document.getElementById("image").files[0];
    if(img) fd.append("image", img);

    const btn = form.querySelector("button[type=submit]");
    const old = btn.textContent;
    btn.disabled = true; btn.textContent = "Generating… (this may take minutes)";
    try{
      const data = await postGenerate(fd);
      if(data.error){ alert(data.error); }
      if(data.video){
        const card = document.createElement("div");
        card.className = "card";
        card.innerHTML = `<video controls src="/${data.video}?t=${Date.now()}"></video>
        <div><small class="mono">${data.meta}</small></div>`;
        document.getElementById("gallery").prepend(card);
      }
    }catch(err){
      alert("Request failed: " + err);
    }finally{
      btn.disabled = false; btn.textContent = old;
    }
  });
}

// ===== Concatenate UI =====
const concatCount = document.getElementById("concat_count");
const concatBuild = document.getElementById("concat_build");
const concatBox   = document.getElementById("concat_prompts");
const concatBtn   = document.getElementById("concat_generate");
const modeFirst   = document.getElementById("mode_first");
const firstImgBox = document.getElementById("first_image_box");
const firstImg    = document.getElementById("image_first");

if(modeFirst){
  modeFirst.addEventListener("change",()=>{
    if(modeFirst.value==="i2v") firstImgBox.classList.remove("hide");
    else firstImgBox.classList.add("hide");
  });
}

function buildConcatFields(){
  const n = Math.max(2, Math.min(20, parseInt(concatCount.value||"2",10)));
  concatBox.innerHTML = "";
  for(let i=0;i<n;i++){
    const div=document.createElement("div");
    div.className="prompt-block";
    div.innerHTML =
      '<label>Prompt '+(i+1)+'</label>' +
      '<textarea id="concat_prompt_'+i+'" rows="2" required></textarea>';
    concatBox.appendChild(div);
  }
}
if(concatBuild) concatBuild.addEventListener("click", buildConcatFields);
// crea subito i campi al load
if(concatBox) buildConcatFields();

async function postConcat(fd){
  const res = await fetch("/generate_concat", {method:"POST", body:fd});
  return await res.json();
}

if(concatBtn){
  concatBtn.addEventListener("click", async ()=>{
    try{
      let n = parseInt(concatCount.value||"2",10);
      if(isNaN(n)) n=2;
      n = Math.max(2, Math.min(20, n));
      const fd = new FormData();
      fd.append("prompt_count", String(n));
      for(let i=0;i<n;i++){
        const t = document.getElementById("concat_prompt_"+i);
        fd.append("prompts", (t && t.value) ? t.value : "");
      }
      fd.append("mode_first", modeFirst.value);

      // Reuse main form params
      fd.append("negative_prompt", document.getElementById("neg").value);
      fd.append("width",  document.getElementById("width").value);
      fd.append("height", document.getElementById("height").value);
      fd.append("frames", document.getElementById("frames").value);
      fd.append("steps",  document.getElementById("steps").value);
      fd.append("cfg",    document.getElementById("cfg").value);
      fd.append("seed",   document.getElementById("seed").value);
      fd.append("strength", document.getElementById("strength").value);
      fd.append("motion_bucket_id", document.getElementById("motion_bucket_id").value);
      fd.append("lora", document.getElementById("lora").value);
      fd.append("lora_scale", document.getElementById("lora_scale").value);
      fd.append("sched", document.getElementById("sched").value);
      fd.append("offload", document.getElementById("offload").checked ? "1":"0");

      if(modeFirst.value==="i2v" && firstImg.files[0]){
        fd.append("image", firstImg.files[0]);
      }

      const old = concatBtn.textContent;
      concatBtn.disabled = true;
      concatBtn.textContent = "Generating...";
      const data = await postConcat(fd);
      concatBtn.disabled = false;
      concatBtn.textContent = old;

      if(data.error){ alert(data.error); return; }
      if(data.final_video){
        const card = document.createElement("div");
        card.className = "card";
        card.innerHTML = `<video controls src="${data.final_video}?t=${Date.now()}"></video>
        <div><small class="mono">Concatenated: ${(data.clips||[]).length} clips</small></div>`;
        document.getElementById("gallery").prepend(card);
      }
    }catch(err){
      concatBtn.disabled = false;
      alert("Concat failed: " + err);
    }
  });
}
