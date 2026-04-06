"""
Sepiru AI — Full-screen premium login gate.
Single-page: HTML canvas background + Streamlit form overlaid via CSS.
"""
import os
import streamlit as st


def _get_password() -> str:
    try:
        p = st.secrets.get("APP_PASSWORD", "")
        if p:
            return p
    except Exception:
        pass
    return os.getenv("APP_PASSWORD", "sepiru")


def require_auth():
    if st.session_state.get("_auth"):
        return

    st.markdown(_full_page_css(), unsafe_allow_html=True)

    # Canvas aurora background
    st.markdown(_canvas_html(), unsafe_allow_html=True)

    # Hero text
    st.markdown("""
    <div class="sep-hero">
        <div class="sep-eyebrow">Data Intelligence Platform</div>
        <div class="sep-name">Sepiru AI</div>
        <div class="sep-tagline">
            Most tools make you work for the answer.
            <span class="sep-accent">Sepiru AI just answers.</span>
        </div>
        <div class="sep-divider"></div>
        <div class="sep-key-label">Enter Access Key</div>
    </div>
    """, unsafe_allow_html=True)

    # Password form — centered, tight
    c1, c2, c3 = st.columns([1, 1.4, 1])
    with c2:
        pwd = st.text_input("key", type="password",
                            placeholder="· · · · · · · ·",
                            label_visibility="collapsed",
                            key="_pw")
        btn = st.button("ENTER →", use_container_width=True, key="_btn")

    if btn:
        if pwd == _get_password():
            st.session_state["_auth"] = True
            st.rerun()
        else:
            _, ec, _ = st.columns([1, 1.4, 1])
            with ec:
                st.markdown("""
                <div class="sep-err">✕ &nbsp;Incorrect access key</div>
                """, unsafe_allow_html=True)

    st.stop()


def _full_page_css() -> str:
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600&family=Cormorant+Garamond:ital,wght@0,300;1,300;1,400&display=swap');

    /* ── Kill all Streamlit chrome ── */
    #MainMenu,footer,header,
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    section[data-testid="stSidebar"] { display:none !important; }

    /* ── Full viewport ── */
    html,body { margin:0!important;padding:0!important;overflow:hidden!important; }
    .stApp { background:#000!important; }
    .main .block-container {
        padding:0!important;
        max-width:100%!important;
        min-height:100vh!important;
        display:flex!important;
        flex-direction:column!important;
        align-items:center!important;
        justify-content:center!important;
    }

    /* ── Hero ── */
    .sep-hero {
        text-align:center;
        position:relative;z-index:10;
        padding:0 1rem;
        animation:sfadeUp .9s cubic-bezier(.22,1,.36,1) both;
    }
    .sep-eyebrow {
        font-family:'Space Grotesk',sans-serif;
        font-size:.6rem;font-weight:500;
        letter-spacing:5px;text-transform:uppercase;
        color:rgba(201,168,76,.5);
        margin-bottom:1rem;
    }
    .sep-name {
        font-family:'Cormorant Garamond',serif;
        font-size:clamp(4.5rem,11vw,8rem);
        font-weight:300;font-style:italic;
        line-height:.9;
        background:linear-gradient(135deg,#fff 0%,#f5e080 25%,#c9a84c 55%,#f5e8b8 80%,#fff 100%);
        background-size:200% auto;
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
        animation:sfadeUp .9s ease .08s both, sgoldFlow 4s linear 1s infinite;
        filter:drop-shadow(0 0 50px rgba(201,168,76,.35));
        margin-bottom:1.2rem;
    }
    .sep-tagline {
        font-family:'Space Grotesk',sans-serif;
        font-size:clamp(.85rem,2vw,1.05rem);
        font-weight:300;
        color:rgba(255,255,255,.45);
        line-height:1.7;
        margin-bottom:0;
    }
    .sep-accent { color:rgba(201,168,76,.85);font-weight:500; }
    .sep-divider {
        width:50px;height:1px;
        background:linear-gradient(90deg,transparent,rgba(201,168,76,.5),transparent);
        margin:.5rem auto 1.4rem;
    }
    .sep-key-label {
        font-family:'Space Grotesk',sans-serif;
        font-size:.55rem;font-weight:500;
        letter-spacing:4px;text-transform:uppercase;
        color:rgba(201,168,76,.35);
        margin-bottom:.5rem;
        animation:sfadeUp .8s ease .3s both;
    }

    /* ── Input ── */
    .stTextInput > div > div > input {
        background:rgba(255,255,255,.04)!important;
        border:1px solid rgba(201,168,76,.25)!important;
        border-radius:12px!important;
        color:#fff!important;
        font-family:'Space Grotesk',sans-serif!important;
        font-size:1.1rem!important;
        letter-spacing:8px!important;
        text-align:center!important;
        padding:.85rem 1rem!important;
        caret-color:#c9a84c!important;
        transition:all .3s ease!important;
    }
    .stTextInput > div > div > input:focus {
        border-color:rgba(201,168,76,.6)!important;
        box-shadow:0 0 0 3px rgba(201,168,76,.09),0 0 40px rgba(201,168,76,.07)!important;
        outline:none!important;
    }
    .stTextInput > div > div > input::placeholder {
        color:rgba(255,255,255,.12)!important;letter-spacing:8px!important;
    }

    /* ── Button ── */
    .stButton > button {
        background:linear-gradient(135deg,#c9a84c 0%,#f5e080 50%,#c9a84c 100%)!important;
        background-size:200% auto!important;
        color:#000!important;border:none!important;
        border-radius:12px!important;
        font-family:'Space Grotesk',sans-serif!important;
        font-size:.7rem!important;font-weight:600!important;
        letter-spacing:3.5px!important;text-transform:uppercase!important;
        padding:.8rem!important;margin-top:.5rem!important;
        box-shadow:0 4px 30px rgba(201,168,76,.28)!important;
        transition:all .3s ease!important;
        animation:sgoldFlow 3s linear infinite!important;
    }
    .stButton > button:hover {
        box-shadow:0 8px 50px rgba(201,168,76,.5)!important;
        transform:translateY(-2px)!important;
    }

    /* ── Error ── */
    .sep-err {
        text-align:center;
        font-family:'Space Grotesk',sans-serif;
        font-size:.78rem;color:#f87171;
        margin-top:.5rem;letter-spacing:.3px;
        padding:.5rem;
        background:rgba(244,63,94,.07);
        border:1px solid rgba(244,63,94,.15);
        border-radius:8px;
    }

    /* ── Keyframes ── */
    @keyframes sfadeUp {
        from{opacity:0;transform:translateY(22px);filter:blur(5px);}
        to{opacity:1;transform:translateY(0);filter:blur(0);}
    }
    @keyframes sgoldFlow {
        0%{background-position:0% center;}
        100%{background-position:200% center;}
    }
    </style>
    """


def _canvas_html() -> str:
    return """
    <canvas id="_sep_cv" style="position:fixed;inset:0;width:100vw;height:100vh;
    pointer-events:none;z-index:0;"></canvas>
    <script>
    (function(){
        const cv=document.getElementById('_sep_cv');
        if(!cv)return;
        const cx=cv.getContext('2d');
        let W,H,pts,mx=9999,my=9999;
        function resize(){W=cv.width=window.innerWidth;H=cv.height=window.innerHeight;}
        resize();window.addEventListener('resize',resize);
        document.addEventListener('mousemove',e=>{mx=e.clientX;my=e.clientY;});
        const BLOBS=[
            {x:.5,y:-.05,rx:.85,ry:.5,h:42, s:80,a:.13,sp:.0002,ph:0},
            {x:.08,y:.9, rx:.5, ry:.35,h:258,s:70,a:.07,sp:.0003,ph:2.1},
            {x:.92,y:.8, rx:.45,ry:.3,h:162,s:65,a:.06,sp:.0004,ph:4.2},
        ];
        function initPts(){
            const N=Math.floor(W*H/16000);
            pts=Array.from({length:N},()=>({
                x:Math.random()*W,y:Math.random()*H,
                vx:(Math.random()-.5)*.22,vy:(Math.random()-.5)*.22,
                r:Math.random()*1.5+.3,
                ph:Math.random()*Math.PI*2,spd:.01+Math.random()*.018,
            }));
        }
        initPts();window.addEventListener('resize',initPts);
        let t=0;
        function draw(){
            requestAnimationFrame(draw);
            cx.clearRect(0,0,W,H);t+=.007;
            BLOBS.forEach(b=>{
                const ox=Math.sin(t*b.sp*1000+b.ph)*.06;
                const oy=Math.cos(t*b.sp*800+b.ph)*.04;
                const px=(b.x+ox)*W,py=(b.y+oy)*H;
                const a=b.a*(.6+Math.sin(t+b.ph)*.4);
                const g=cx.createRadialGradient(px,py,0,px,py,Math.max(W,H)*b.rx);
                g.addColorStop(0,`hsla(${b.h},${b.s}%,60%,${a})`);
                g.addColorStop(.5,`hsla(${b.h},${b.s}%,40%,${a*.4})`);
                g.addColorStop(1,'transparent');
                cx.fillStyle=g;cx.fillRect(0,0,W,H);
            });
            for(let i=0;i<pts.length;i++){
                const a=pts[i];
                for(let j=i+1;j<pts.length;j++){
                    const b=pts[j];
                    const dx=a.x-b.x,dy=a.y-b.y,d2=dx*dx+dy*dy;
                    if(d2<110*110){
                        cx.beginPath();
                        cx.strokeStyle=`rgba(201,168,76,${.09*(1-Math.sqrt(d2)/110)})`;
                        cx.lineWidth=.35;cx.moveTo(a.x,a.y);cx.lineTo(b.x,b.y);cx.stroke();
                    }
                }
            }
            for(const p of pts){
                p.ph+=p.spd;
                const g=.4+Math.sin(p.ph)*.28;
                const dx=p.x-mx,dy=p.y-my,d2=dx*dx+dy*dy;
                if(d2<8100){const d=Math.sqrt(d2);const f=(90-d)/90*.38;p.vx+=dx/d*f;p.vy+=dy/d*f;}
                p.vx*=.992;p.vy*=.992;
                cx.beginPath();cx.arc(p.x,p.y,p.r,0,Math.PI*2);
                cx.fillStyle=`rgba(201,168,76,${g})`;cx.fill();
                p.x+=p.vx;p.y+=p.vy;
                if(p.x<-10)p.x=W+10;if(p.x>W+10)p.x=-10;
                if(p.y<-10)p.y=H+10;if(p.y>H+10)p.y=-10;
            }
        }
        draw();
    })();
    </script>
    """
