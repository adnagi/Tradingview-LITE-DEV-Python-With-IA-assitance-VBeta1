import os
import json
import requests
import math
from flask import Flask, render_template_string, request, jsonify
from openai import OpenAI

app = Flask(__name__)

MEMORY_FILE = "coach_memory.dat"
# Création du fichier au démarrage s'il n'existe pas
if not os.path.exists(MEMORY_FILE):
    open(MEMORY_FILE, 'w', encoding='utf-8').close()

def get_coach_memory_summary():
    """Lit l'historique et génère un résumé des erreurs pour le prompt IA"""
    try:
        with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        trades = [json.loads(line) for line in lines if line.strip()]
        if not trades:
            return "\n\n[MÉMOIRE DU COACH] Aucun historique de trade pour le moment."

        total = len(trades)
        wins = len([t for t in trades if t.get('result') == 'WIN'])
        win_rate = (wins / total) * 100

        losses = [t for t in trades if t.get('result') == 'LOSS']
        recent_losses = losses[-3:] # On prend les 3 dernières erreurs

        summary = f"\n\n[MÉMOIRE DU COACH] Historique global : {total} trades analysés (Taux de réussite: {win_rate:.1f}%).\n"
        if recent_losses:
            summary += "IMPORTANT - Apprends de tes récentes ERREURS (Stop Loss touchés) :\n"
            for idx, l in enumerate(recent_losses):
                summary += f"- Erreur sur un setup {l.get('type')} basé sur '{l.get('reason')}'.\n"
            summary += "Adapte ton analyse actuelle pour être beaucoup plus strict si tu vois des conditions similaires à ces erreurs.\n"
        return summary
    except Exception as e:
        return f"\n[Erreur de lecture de la mémoire: {str(e)}]"


KEYS_FILE = "api_keys.json"

def load_keys():
    if os.path.exists(KEYS_FILE):
        with open(KEYS_FILE, 'r') as f:
            return json.load(f)
    return {"deepseek_key": "", "grok_key": "", "deepseek_model": "deepseek-chat", "grok_model": "grok-beta"}

def save_keys(keys):
    with open(KEYS_FILE, 'w') as f:
        json.dump(keys, f)

@app.route('/api/keys', methods=['GET', 'POST'])
def manage_keys():
    if request.method == 'POST':
        save_keys(request.json)
        return jsonify({"status": "success"})
    return jsonify(load_keys())

@app.route('/api/analyze', methods=['POST'])
def analyze_chart():
    data = request.json
    ai_type = data.get('ai_type')
    context = data.get('context', '')
    strategy = data.get('strategy', 'SMC Multi-Stratégies')
    keys = load_keys()
    
    if "Builder Modulaire" in strategy:
        system_prompt = f"""Tu es un Ingénieur Quantitatif. Modèle : {strategy}.
Analyse les confluences mathématiques choisies par l'utilisateur (Algo, Variance, L2, Optimisation, Convexe, Linéaire).
Réponds UNIQUEMENT avec ce format HTML :
<div style="color:white;">
    <div style="margin-bottom:8px;">🎯 <b>Verdict Custom :</b> [Attendre / Entrer LONG / Entrer SHORT]</div>
    <div style="margin-bottom:8px;">🟢 <b>Entrée Opti. :</b> [Prix]</div>
    <div style="margin-bottom:8px;">🔴 <b>Stop Loss :</b> [Prix]</div>
    <div style="margin-bottom:8px;">💰 <b>Take Profit :</b> [Prix projeté]</div>
    <div style="margin-top:10px; padding-top:10px; border-top:1px solid #4a5056; font-style:italic; color:#fdd835;">
        🧪 <b>Analyse des Confluences :</b> [Explique pourquoi les briques sélectionnées s'alignent].
    </div>
</div>"""
    elif "Quant V2" in strategy:
        system_prompt = f"""Tu es un Lead Quant Researcher. Modèle : {strategy}.
Analyse les données du modèle V2 (Filtre de Kalman, Lambda Dynamique ATR, Convexité 2nd degré).
Réponds UNIQUEMENT avec ce format HTML :
<div style="color:white;">
    <div style="margin-bottom:8px;">🎯 <b>Verdict Quant V2 :</b> [Attendre / Entrer LONG / Entrer SHORT]</div>
    <div style="margin-bottom:8px;">🟢 <b>Entrée Opti. :</b> [Prix]</div>
    <div style="margin-bottom:8px;">🔴 <b>Stop Loss :</b> [Prix]</div>
    <div style="margin-bottom:8px;">💰 <b>Take Profit :</b> [Prix projeté]</div>
    <div style="margin-top:10px; padding-top:10px; border-top:1px solid #4a5056; font-style:italic; color:#00e5ff;">
        🧬 <b>Thèse Algorithmique :</b> [Explique la probabilité et la convexité polynomiale].
    </div>
</div>"""
    elif "Quant" in strategy:
        system_prompt = f"""Tu es un Ingénieur Quantitatif. Modèle ciblé : {strategy}.
Analyse les probabilités (Régression L2, Optimisation Convexe, POC).
Réponds UNIQUEMENT avec ce format HTML :
<div style="color:white;">
    <div style="margin-bottom:8px;">🎯 <b>Verdict Quant :</b> [Attendre / Entrer LONG / Entrer SHORT]</div>
    <div style="margin-bottom:8px;">🟢 <b>Entrée Opti. :</b> [Prix]</div>
    <div style="margin-bottom:8px;">🔴 <b>Stop Loss :</b> [Prix]</div>
    <div style="margin-bottom:8px;">💰 <b>Take Profit :</b> [Prix]</div>
    <div style="margin-top:10px; padding-top:10px; border-top:1px solid #4a5056; font-style:italic; color:#e040fb;">
        🧠 <b>Thèse Mathématique :</b> [Explique la convergence L2 et POC].
    </div>
</div>"""
    else:
        system_prompt = f"""Tu es un Coach de Trading SMC. Stratégies ciblées : {strategy}.
Analyse les confluences fournies. Réponds UNIQUEMENT avec ce format HTML :
<div style="color:white;">
    <div style="margin-bottom:8px;">🎯 <b>Verdict IA :</b> [Attendre / Entrer LONG / Entrer SHORT]</div>
    <div style="margin-bottom:8px;">🟢 <b>Entrée :</b> [Prix]</div>
    <div style="margin-bottom:8px;">🔴 <b>Stop Loss :</b> [Prix]</div>
    <div style="margin-bottom:8px;">💰 <b>Take Profit :</b> [Prix]</div>
    <div style="margin-top:10px; padding-top:10px; border-top:1px solid #4a5056; font-style:italic; color:#ff9800;">
        🧠 <b>Analyse Institutionnelle :</b> [Explique les confluences : OB, FVG, Liquidity].
    </div>
</div>"""

    try:
        # INJECTION DE LA MÉMOIRE DANS LE PROMPT
        memory_summary = get_coach_memory_summary()
        system_prompt += memory_summary

        if ai_type == 'deepseek':
            if not keys.get('deepseek_key'): return jsonify({"error": "Clé Deepseek manquante."}), 400
            client = OpenAI(api_key=keys.get('deepseek_key'), base_url="https://api.deepseek.com")
            response = client.chat.completions.create(model=keys.get('deepseek_model', 'deepseek-chat'), messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": context}], stream=False, max_tokens=450, temperature=0.3)
            return jsonify({"response": response.choices[0].message.content})
        elif ai_type == 'grok':
            if not keys.get('grok_key'): return jsonify({"error": "Clé Grok manquante."}), 400
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {keys.get('grok_key')}"}
            payload = {"model": keys.get('grok_model', 'grok-beta'), "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": context}], "stream": False, "temperature": 0.3}
            grok_req = requests.post("https://api.x.ai/v1/chat/completions", headers=headers, json=payload)
            grok_res = grok_req.json()
            if grok_req.status_code == 200: return jsonify({"response": grok_res['choices'][0]['message']['content']})
            else: return jsonify({"error": f"Erreur Grok : {grok_res.get('error', 'Erreur')}"}), 400
        else: return jsonify({"error": "Type d'IA invalide"}), 400
    except Exception as e: return jsonify({"error": f"Erreur Serveur : {str(e)}"}), 500



@app.route('/api/sl_params', methods=['GET'])
def optimize_sl():
    try:
        # VERIFICATION: S'assurer que le fichier existe
        if not os.path.exists(MEMORY_FILE):
            open(MEMORY_FILE, 'w', encoding='utf-8').close()

        with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        trades = [json.loads(line) for line in lines if line.strip()]

        atr_mult = 2.0
        rr_target = 2.0

        if len(trades) > 3:
            losses = [t for t in trades if t.get('result') == 'LOSS']
            win_rate = (len(trades) - len(losses)) / len(trades)

            # Intelligence du coach : Si trop de Stop Loss touchés (win rate bas), on éloigne le SL
            if win_rate < 0.40:
                atr_mult = 3.0  # On donne plus d'espace au trade
            # Si très précis, on peut resserrer pour un meilleur ratio
            elif win_rate > 0.65:
                atr_mult = 1.5
            else:
                atr_mult = 2.2

        return jsonify({"atr_multiplier": atr_mult, "rr_target": rr_target, "trades_count": len(trades)})
    except Exception as e:
        return jsonify({"atr_multiplier": 2.0, "rr_target": 2.0, "error": str(e)})

@app.route('/api/learn', methods=['POST'])
def learn_from_trade():
    data = request.json
    try:
        with open(MEMORY_FILE, "a", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            f.write("\n")
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SMC & Quant Coach Pro Terminal</title>
    <script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        body { font-family: Arial, sans-serif; background-color: #131722; color: white; margin: 0; padding: 20px; display: flex; flex-direction: column; height: 100vh; box-sizing: border-box; }
        .header-container { display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px; flex-wrap: wrap; gap: 15px; flex-shrink: 0;}
        .crypto-selector select { background-color: #2b3139; color: white; border: 1px solid #4a5056; padding: 10px 15px; font-size: 18px; border-radius: 4px; cursor: pointer; outline: none; font-weight:bold;}
        .live-price-container { display: flex; align-items: baseline; gap: 10px; }
        .live-price-label { font-size: 16px; color: #848e9c; }
        #live-price { font-size: 32px; font-weight: bold; font-family: 'Courier New', Courier, monospace; }
        
        .toolbar { display: flex; gap: 8px; margin-bottom: 10px; flex-wrap: wrap; align-items: center; flex-shrink: 0; width: 100%;}
        button { background-color: #2b3139; color: white; border: 1px solid #4a5056; padding: 8px 16px; cursor: pointer; border-radius: 4px; }
        button:hover { background-color: #3b4249; }
        button.active { background-color: #fcd535; color: black; font-weight: bold; }

        .indicators-panel { display: flex; gap: 15px; margin-left: auto; align-items: center; background: #1e222d; padding: 5px 15px; border-radius: 8px; border: 1px solid #2B2B43;}
        .switch-wrapper { display: flex; align-items: center; gap: 6px; font-size: 13px; font-weight: bold; }
        .switch { position: relative; display: inline-block; width: 34px; height: 20px; }
        .switch input { opacity: 0; width: 0; height: 0; }
        .slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: #4a5056; transition: .4s; border-radius: 20px; }
        .slider:before { position: absolute; content: ""; height: 14px; width: 14px; left: 3px; bottom: 3px; background-color: white; transition: .4s; border-radius: 50%; }
        input:checked + .slider { background-color: #26a69a; }
        input:checked + .slider.orange { background-color: #ff9800; }
        input:checked + .slider.cyan { background-color: #00bcd4; }
        input:checked + .slider.purple { background-color: #9c27b0; }
        input:checked + .slider.blue { background-color: #00e5ff; }
        input:checked + .slider.yellow { background-color: #fdd835; }
        input:checked + .slider.coach { background-color: #e91e63; }
        input:checked + .slider:before { transform: translateX(14px); }

        .dropdown { position: relative; display: inline-block; margin-left: 10px; }
        .dropdown-btn { background-color: #2196f3; color: white; border: 1px solid #1e88e5; padding: 8px 20px; font-size: 14px; font-weight: bold; border-radius: 6px; cursor: pointer; display: flex; align-items: center; gap: 8px;}
        .dropdown-btn:hover { background-color: #1976d2; }
        .dropdown-content { display: none; position: absolute; right: 0; background-color: #1e222d; min-width: 320px; box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.5); border-radius: 6px; border: 1px solid #4a5056; z-index: 500; margin-top: 5px; padding: 10px 0;}
        .dropdown-content.show { display: block; }
        .menu-item { padding: 10px 20px; display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid rgba(74, 80, 86, 0.3); }

        .charts-wrapper { display: flex; flex-direction: column; flex-grow: 1; min-height: 0; border: 1px solid #2B2B43; position: relative;}
        #main-chart-wrapper { position: relative; flex-grow: 1; min-height: 300px; }
        #chart-container { width: 100%; height: 100%; position: absolute; top: 0; left: 0;}
        #ob-overlay { position: absolute; top: 0; left: 0; pointer-events: none; z-index: 100; width: 100%; height: 100%;}
        .resizer { height: 10px; background-color: #2b3139; cursor: row-resize; display: flex; justify-content: center; align-items: center; z-index: 20;}
        .resizer::after { content: "|||"; color: #848e9c; font-size: 8px; transform: rotate(90deg); }
        #volume-container { height: 150px; width: 100%; position: relative; min-height: 50px; flex-shrink: 0; border-top: 1px solid #2B2B43;}
        
        #coach-hud { position: absolute; top: 10px; left: 10px; z-index: 200; background: rgba(19, 23, 34, 0.95); padding: 20px; border-radius: 12px; border: 2px solid #e91e63; display: none; font-size: 14px; width: 400px; box-shadow: 0px 8px 32px rgba(233,30,99,0.3); backdrop-filter: blur(10px);}
        .coach-title { font-weight: bold; color: #fff; margin-bottom: 15px; display: flex; align-items: center; justify-content: space-between; font-size:18px; border-bottom: 1px solid rgba(233,30,99,0.5); padding-bottom: 10px;}
        .coach-title .badge { background: #e91e63; padding: 3px 8px; border-radius: 4px; font-size: 12px; text-transform: uppercase; letter-spacing: 1px;}
        
        .active-strats-tags { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 10px; }
        .strat-tag { font-size: 10px; padding: 3px 6px; border-radius: 4px; background: #2b3139; color: #848e9c; font-weight: bold;}
        .strat-tag.active { color: white; border: 1px solid;}
        
        .scanner-status { background: #2b3139; border: 1px solid #4caf50; border-radius: 6px; padding: 10px; margin-bottom: 15px; text-align: center; font-weight: bold; color: #4caf50; display: flex; align-items: center; justify-content: center; gap: 10px; animation: pulse-border 2s infinite; }
        @keyframes pulse-border { 0% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.4); } 70% { box-shadow: 0 0 0 8px rgba(76, 175, 80, 0); } 100% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0); } }
        
        .ai-buttons { display: flex; gap: 8px; margin-bottom: 15px; }
        .ai-btn { padding: 10px 12px; font-size: 13px; border-radius: 6px; border: none; cursor: pointer; font-weight: bold; flex: 1; transition: 0.2s;}
        .ai-btn.deepseek { background-color: #2b3139; color: white; border: 1px solid #ffffff; }
        .ai-btn.grok { background-color: #2b3139; color: #1da1f2; border: 1px solid #1da1f2; }
        .ai-btn:hover { background-color: #4a5056; }
        
        #ai-response { font-size: 14px; line-height: 1.6; background: #1e222d; padding: 15px; border-radius: 8px; border: 1px solid #4a5056; min-height: 80px;}

        .modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.8); backdrop-filter: blur(5px); }
        .modal-content { background-color: #1e222d; margin: 5% auto; padding: 30px; border: 1px solid #4a5056; width: 450px; border-radius: 12px; box-shadow: 0px 10px 40px rgba(0,0,0,0.8);}
        .modal-content input[type="text"], .modal-content input[type="password"] { width: 100%; padding: 12px; margin: 8px 0 20px 0; background: #131722; border: 1px solid #4a5056; color: white; border-radius: 6px; box-sizing: border-box; font-family: monospace;}
        .modal-content label { font-size: 13px; color: #848e9c; font-weight: bold; text-transform: uppercase; }
        .close-modal { color: #aaa; float: right; font-size: 28px; font-weight: bold; cursor: pointer; margin-top: -10px;}
        .close-modal:hover { color: white; }
    </style>
</head>
<body>

    <div class="header-container">
        <div class="crypto-selector">
            <select id="symbol-select">
                <option value="BTCUSDT" selected>Bitcoin (BTC/USDT)</option>
                <option value="ETHUSDT">Ethereum (ETH/USDT)</option>
                <option value="SOLUSDT">Solana (SOL/USDT)</option>
                <option value="XRPUSDT">Ripple (XRP/USDT)</option>
                <option value="DOGEUSDT">Dogecoin (DOGE/USDT)</option>
            </select>
        </div>
        <div class="live-price-container">
            <span class="live-price-label">Prix Actuel :</span>
            <span id="live-price" style="color: #fcd535;">Connexion en cours...</span>
        </div>
    </div>
    
    <div class="toolbar" id="timeframes">
        <button data-interval="1m" class="active">1m</button>
        <button data-interval="3m">3m</button>
        <button data-interval="5m">5m</button>
        <button data-interval="15m">15m</button>
        <button data-interval="30m">30m</button>
        <button data-interval="1h">1H</button>
        <button data-interval="4h">4H</button>

        <div class="indicators-panel">
            <button id="settings-btn" style="background:transparent; border:none; font-size:18px; cursor:pointer;" title="Paramètres IA">⚙️</button>
            <div class="switch-wrapper" style="margin-right: 15px; border-right: 1px solid #4a5056; padding-right: 15px;">
                <span style="color: #e91e63;">🧠 COACH PRO</span>
                <label class="switch"><input type="checkbox" id="toggle-mentor"><span class="slider coach"></span></label>
            </div>
            
            <div class="switch-wrapper"><span>Liq. (BSL/SSL)</span><label class="switch"><input type="checkbox" id="toggle-liq"><span class="slider yellow"></span></label></div>
            <div class="switch-wrapper"><span>FVG</span><label class="switch"><input type="checkbox" id="toggle-fvg"><span class="slider"></span></label></div>
            <div class="switch-wrapper"><span>OB</span><label class="switch"><input type="checkbox" id="toggle-ob-valid"><span class="slider orange"></span></label></div>
        </div>

        <div class="dropdown">
            <button class="dropdown-btn" id="dropdown-btn">📈 Stratégies Auto ▼</button>
            <div class="dropdown-content" id="dropdown-menu">
                <div class="menu-item">
                    <span style="font-size: 13px; font-weight: bold; color: #ff9800;">🎯 SMC : Scanner OB</span>
                    <label class="switch"><input type="checkbox" name="strat" id="toggle-strat-ob" checked><span class="slider orange"></span></label>
                </div>
                <div class="menu-item">
                    <span style="font-size: 13px; font-weight: bold; color: #00bcd4;">💧 SMC : Liquidity Sweep</span>
                    <label class="switch"><input type="checkbox" name="strat" id="toggle-strat-liq"><span class="slider cyan"></span></label>
                </div>
                <div class="menu-item" style="background-color: rgba(156, 39, 176, 0.1); border-top: 1px solid #4a5056; padding-top: 15px;">
                    <span style="font-size: 13px; font-weight: bold; color: #e040fb;">⚛️ Quant V1 (L2 Statique)</span>
                    <label class="switch"><input type="checkbox" name="strat" id="toggle-strat-quant"><span class="slider purple"></span></label>
                </div>
                <div class="menu-item" style="background-color: rgba(0, 229, 255, 0.1); border-top: 1px solid #4a5056; padding-top: 15px;">
                    <span style="font-size: 13px; font-weight: bold; color: #00e5ff;">🧠 Quant V2 (Pro Dynamique)</span>
                    <label class="switch"><input type="checkbox" name="strat" id="toggle-strat-quant-v2"><span class="slider blue"></span></label>
                </div>
                <!-- NOUVELLE STRATÉGIE MODULAIRE -->
                <div class="menu-item" style="background-color: rgba(253, 216, 53, 0.1); border-top: 1px solid #4a5056; padding-top: 15px; flex-direction: column; align-items: stretch;">
                    <div style="display: flex; justify-content: space-between; align-items: center; width: 100%;">
                        <span style="font-size: 13px; font-weight: bold; color: #fdd835;">🧪 Builder Modulaire Custom</span>
                        <label class="switch"><input type="checkbox" name="strat" id="toggle-strat-builder"><span class="slider yellow"></span></label>
                    </div>
                    <div id="builder-options" style="display:none; margin-top: 10px; padding: 12px; background: rgba(0,0,0,0.3); border-radius: 6px; border-left: 3px solid #fdd835;">
                        <div style="display:flex; justify-content:space-between; margin-bottom:8px; font-size:12px;"><span>Algorithme</span><input type="checkbox" id="sub-algo" class="sub-strat" checked></div>
                        <div style="display:flex; justify-content:space-between; margin-bottom:8px; font-size:12px;"><span>Moyenne Variance</span><input type="checkbox" id="sub-meanvar" class="sub-strat"></div>
                        <div style="display:flex; justify-content:space-between; margin-bottom:8px; font-size:12px;"><span>Régression L2</span><input type="checkbox" id="sub-l2" class="sub-strat"></div>
                        <div style="display:flex; justify-content:space-between; margin-bottom:8px; font-size:12px;"><span>Optimisation</span><input type="checkbox" id="sub-opti" class="sub-strat"></div>
                        <div style="display:flex; justify-content:space-between; margin-bottom:8px; font-size:12px;"><span>Convexe</span><input type="checkbox" id="sub-convex" class="sub-strat"></div>
                        <div style="display:flex; justify-content:space-between; font-size:12px;"><span>Régression Linéaire</span><input type="checkbox" id="sub-lr" class="sub-strat"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="charts-wrapper">
        <div id="main-chart-wrapper">
            <div id="coach-hud">
                <div class="coach-title">
                    <span id="hud-title">Mentor Institutionnel</span>
                    <span class="badge" id="hud-badge">Scanner Multi-Strat</span>
                </div>
                
                <div class="active-strats-tags" id="active-strats-tags">
                    <span class="strat-tag active" style="border-color:#ff9800; background:rgba(255,152,0,0.2);">📡 Focus : Order Blocks</span>
                </div>

                <div id="scanner-status" class="scanner-status">
                    <span class="pulse-dot" style="height:10px; width:10px; background-color:#4caf50; border-radius:50%; display:inline-block;"></span>
                    Analyse en cours...
                </div>
                <div class="ai-buttons">
                    <button class="ai-btn deepseek" onclick="askAI('deepseek')">🐋 Analyser avec Deepseek</button>
                    <button class="ai-btn grok" onclick="askAI('grok')">✖️ Analyser avec Grok</button>
                </div>
                <div id="ai-response">
                    <span style="color:#848e9c; font-style:italic;">Le Coach scanne automatiquement le graphique. Dès qu'un Setup fort est détecté, l'alerte visuelle s'activera.</span>
                </div>
            </div>
            
            <div id="chart-container"></div>
            <canvas id="ob-overlay"></canvas>
        </div>
        <div class="resizer" id="resizer"></div>
        <div id="volume-container"></div>
    </div>

    <!-- MODAL -->
    <div id="api-modal" class="modal">
        <div class="modal-content">
            <span class="close-modal">&times;</span>
            <h3 style="color:white; margin-top:0; border-bottom:1px solid #4a5056; padding-bottom:10px;">⚙️ Configurer l'IA Externe</h3>
            <div style="border-left: 3px solid #fff; padding-left: 15px; margin-bottom: 25px; margin-top: 15px;">
                <label>🐋 Clé API Deepseek :</label>
                <input type="password" id="key-deepseek" placeholder="sk-...">
                <label>Modèle Deepseek :</label>
                <input type="text" id="model-deepseek" value="deepseek-chat">
            </div>
            <div style="border-left: 3px solid #1da1f2; padding-left: 15px;">
                <label>✖️ Clé API Grok (xAI) :</label>
                <input type="password" id="key-grok" placeholder="xai-...">
                <label>Modèle Grok :</label>
                <input type="text" id="model-grok" value="grok-beta">
            </div>
            <button onclick="saveApiKeys()" style="width:100%; margin-top:20px; padding:15px; background-color:#26a69a; color:white; border:none; border-radius:6px; font-weight:bold; cursor:pointer; font-size:16px;">Sauvegarder</button>
            <div id="api-save-msg" style="color:#26a69a; margin-top:15px; font-size:14px; display:none; text-align:center; font-weight:bold;">✅ Sauvegardé avec succès !</div>
        </div>
    </div>

    <script>
        const apiModal = document.getElementById('api-modal');
        document.getElementById('settings-btn').addEventListener('click', () => {
            fetch('/api/keys').then(res => res.json()).then(data => {
                document.getElementById('key-deepseek').value = data.deepseek_key || ''; document.getElementById('model-deepseek').value = data.deepseek_model || 'deepseek-chat';
                document.getElementById('key-grok').value = data.grok_key || ''; document.getElementById('model-grok').value = data.grok_model || 'grok-beta';
                apiModal.style.display = 'block';
            });
        });
        document.querySelector('.close-modal').addEventListener('click', () => apiModal.style.display = 'none');
        function saveApiKeys() {
            const payload = { deepseek_key: document.getElementById('key-deepseek').value, deepseek_model: document.getElementById('model-deepseek').value, grok_key: document.getElementById('key-grok').value, grok_model: document.getElementById('model-grok').value };
            fetch('/api/keys', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) }).then(() => {
                document.getElementById('api-save-msg').style.display = 'block'; setTimeout(() => { document.getElementById('api-save-msg').style.display = 'none'; apiModal.style.display = 'none'; }, 1500);
            });
        }

        let quantData = { ridgeProj: 0, ridgeSlope: 0, poc: 0, bbUpper: 0, bbLower: 0, supertrend: 0, v2Score: 0, v2Convexity: 0 };

        function calculateEMA(data, period) {
            if(data.length < period) return data[data.length-1].close;
            let k = 2 / (period + 1); let ema = data[0].close;
            for(let i=1; i<data.length; i++) ema = (data[i].close * k) + (ema * (1-k));
            return ema;
        }

        function calculateRSI(data, period=14) {
            if (data.length < period + 1) return 50;
            let gains = 0, losses = 0;
            for(let i = data.length - period; i < data.length; i++) {
                let diff = data[i].close - data[i-1].close;
                if(diff > 0) gains += diff; else losses -= diff;
            }
            let avgGain = gains / period; let avgLoss = losses / period;
            if (avgLoss === 0) return 100;
            let rs = avgGain / avgLoss; return 100 - (100 / (1 + rs));
        }

        function calculateRidgeRegression(data, period=20, lambda=1.0) {
            if(!data || data.length < period) return { slope: 0, projected: 0 };
            let sumX=0, sumY=0, sumXY=0, sumXX=0;
            for(let i=0; i<period; i++) {
                let x = i; let y = data[data.length - period + i].close;
                sumX += x; sumY += y; sumXY += x*y; sumXX += x*x;
            }
            let meanX = sumX/period; let meanY = sumY/period;
            let numerator=0, denominator=0;
            for(let i=0; i<period; i++) {
                let x = i - meanX; let y = data[data.length - period + i].close - meanY;
                numerator += x*y; denominator += x*x;
            }
            let slope = numerator / (denominator + lambda); 
            let intercept = meanY - slope * meanX;
            return { slope: slope, projected: slope * period + intercept };
        }

        function calculatePOC(data, period=50) {
            if(!data || data.length < period) return 0;
            let profile = {};
            for(let i=data.length-period; i<data.length; i++) {
                let price = Math.round(data[i].close * 100) / 100; 
                profile[price] = (profile[price] || 0) + data[i].volume;
            }
            let pocPrice = 0; let maxVol = 0;
            for (let p in profile) { if(profile[p] > maxVol) { maxVol = profile[p]; pocPrice = parseFloat(p); } }
            return pocPrice;
        }

        function calculateVolIndicators(data) {
            if(!data || data.length < 20) return;
            let sum = 0; for(let i=data.length-20; i<data.length; i++) sum += data[i].close;
            let sma = sum/20; let variance = 0;
            for(let i=data.length-20; i<data.length; i++) variance += Math.pow(data[i].close - sma, 2);
            let sd = Math.sqrt(variance/20);
            quantData.bbUpper = sma + (2*sd); quantData.bbLower = sma - (2*sd);
            
            let trSum = 0;
            for(let i=data.length-14; i<data.length; i++) {
                let curr = data[i], prev = data[i-1];
                trSum += Math.max(curr.high - curr.low, Math.abs(curr.high - prev.close), Math.abs(curr.low - prev.close));
            }
            let atr = trSum / 14;
            quantData.supertrend = data[data.length-1].close > sma ? sma - (atr*3) : sma + (atr*3);
            return atr;
        }

        async function askAI(aiType) {
            const responseDiv = document.getElementById('ai-response');
            let data = window.globalCandles;
            if(!data || data.length === 0) return;
            
            document.getElementById('scanner-status').innerText = "Appel IA Externe... ⏳"; document.getElementById('scanner-status').style.borderColor = "#fcd535"; document.getElementById('scanner-status').style.color = "#fcd535";
            
            let oldHTML = responseDiv.innerHTML;
            responseDiv.innerHTML = '<div style="text-align:center; padding:15px; border-bottom:1px solid #4a5056; margin-bottom:15px;">Envoi des données aux serveurs IA...</div>' + oldHTML;
            
            let currentPrice = data[data.length - 1].close;
            let strats = [];
            if(opts.stratOb) strats.push("Order Blocks");
            if(opts.stratLiq) strats.push("Liquidity Sweeps");
            if(opts.stratQuant) strats.push("Quant V1");
            if(opts.stratQuantV2) strats.push("Quant V2");
            if(opts.stratBuilder) strats.push("Builder Modulaire Custom");

            let dataContext = "Actif: " + currentSymbol + " | TF: " + currentInterval + " | Prix: " + currentPrice + "\\nStratégies actives: " + strats.join(' + ');

            if(activeSetup) dataContext += "\\nSetup local détecté : " + activeSetup.type + " avec Raison: " + activeSetup.reason;

            try {
                const res = await fetch('/api/analyze', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ ai_type: aiType, context: dataContext, strategy: strats.join(' + ') }) });
                const d = await res.json();
                document.getElementById('scanner-status').innerText = "Avis IA Reçu ✓"; document.getElementById('scanner-status').style.color = "#2196f3"; document.getElementById('scanner-status').style.borderColor = "#2196f3";
                responseDiv.innerHTML = (d.error ? "<span style='color:#ef5350;'>❌ " + d.error + "</span>" : d.response) + "<hr style='border:none; border-top:1px dashed #4a5056; margin:15px 0;'>" + oldHTML;
            } catch(e) { 
                responseDiv.innerHTML = "<span style='color:#ef5350; padding:10px; display:block;'>❌ Erreur réseau IA.</span>" + oldHTML; 
            }
        }

        const dropdownBtn = document.getElementById('dropdown-btn'); const dropdownMenu = document.getElementById('dropdown-menu');
        dropdownBtn.addEventListener('click', () => dropdownMenu.classList.toggle('show'));
        window.addEventListener('click', (event) => { 
            // On empêche la fermeture si on clique dans le menu pour cocher les cases sub-stratégies
            if (!event.target.matches('.dropdown-btn') && !event.target.closest('.dropdown-content')) {
                dropdownMenu.classList.remove('show'); 
            }
        });

        const priceElement = document.getElementById('live-price');
        const coachHud = document.getElementById('coach-hud');
        const tagsContainer = document.getElementById('active-strats-tags');
        
        let currentSymbol = 'BTCUSDT'; let currentInterval = '1m'; let previousPrice = 0;
        window.globalCandles = []; let validOBsList = []; let fvgList = []; let liquidityList = []; 
        let opts = { fvg: false, obValid: false, liq: false, stratOb: true, stratLiq: false, stratQuant: false, stratQuantV2: false, stratBuilder: false, mentor: false,
                     subAlgo: true, subMeanVar: false, subL2: false, subOpti: false, subConvex: false, subLr: false };

        let activeSetup = null; let setupLines = [];

        window.clearActiveSetup = function() {
            updateCoachMemoryParams(); // Le coach réévalue ses paramètres après un trade
            if (setupLines && setupLines.length > 0) { setupLines.forEach(l => { try { candleSeries.removePriceLine(l); } catch(e) {} }); }
            setupLines = []; activeSetup = null;
            document.getElementById('scanner-status').innerHTML = '<span class="pulse-dot" style="height:10px; width:10px; background-color:#4caf50; border-radius:50%; display:inline-block;"></span> Analyse en cours...';
            document.getElementById('scanner-status').style.borderColor = "#4caf50";
            document.getElementById('ai-response').innerHTML = "<span style='color:#848e9c; font-style:italic;'>Le Coach scanne automatiquement le graphique. Dès qu'un Setup fort est détecté, l'alerte visuelle s'activera.</span>";
        };

        function updatePriceDisplay(newPrice) {
            if(!newPrice) return;
            const decimals = newPrice < 10 ? 4 : 2;
            priceElement.innerText = "$" + newPrice.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
            if (newPrice > previousPrice) priceElement.style.color = '#26a69a'; else if (newPrice < previousPrice) priceElement.style.color = '#ef5350'; else priceElement.style.color = 'white';
            previousPrice = newPrice;
        }

        function updateTags() {
            let html = '';
            if (opts.stratOb) html += '<span class="strat-tag active" style="border-color:#ff9800; background:rgba(255,152,0,0.2);">📡 OB SMC</span>';
            if (opts.stratLiq) html += '<span class="strat-tag active" style="border-color:#00bcd4; background:rgba(0,188,212,0.2);">💧 Liq. Sweep</span>';
            
            if (opts.stratBuilder) {
                html += '<span class="strat-tag active" style="border-color:#fdd835; background:rgba(253, 216, 53, 0.2); color:#fdd835;">🧪 Builder Custom</span>';
                document.getElementById('hud-title').innerText = "Laboratoire Modulaire";
                document.getElementById('hud-badge').style.background = "#fdd835"; document.getElementById('hud-badge').style.color = "#000"; document.getElementById('hud-badge').innerText = "MULTI-SIGNAUX";
            } else if (opts.stratQuantV2) {
                html += '<span class="strat-tag active" style="border-color:#00e5ff; background:rgba(0, 229, 255, 0.2); color:#00e5ff;">🧠 Quant V2 (Pro)</span>';
                document.getElementById('hud-title').innerText = "Quant V2 (Kalman & L2)";
                document.getElementById('hud-badge').style.background = "#00e5ff"; document.getElementById('hud-badge').style.color = "#000"; document.getElementById('hud-badge').innerText = "SCANNER IA";
            } else if (opts.stratQuant) {
                html += '<span class="strat-tag active" style="border-color:#e040fb; background:rgba(156, 39, 176, 0.2);">⚛️ Quant V1 (L2)</span>';
                document.getElementById('hud-title').innerText = "Scanner Quantitatif L2";
                document.getElementById('hud-badge').style.background = "#e040fb"; document.getElementById('hud-badge').style.color = "#fff"; document.getElementById('hud-badge').innerText = "MATH SIGNAUX";
            } else {
                document.getElementById('hud-title').innerText = "Mentor Institutionnel";
                document.getElementById('hud-badge').style.background = "#e91e63"; document.getElementById('hud-badge').style.color = "#fff"; document.getElementById('hud-badge').innerText = "SCANNER SMC";
            }
            if (!opts.stratOb && !opts.stratLiq && !opts.stratQuant && !opts.stratQuantV2 && !opts.stratBuilder) html += '<span class="strat-tag">Aucune stratégie</span>';
            tagsContainer.innerHTML = html;
        }

        const commonOptions = { layout: { textColor: '#d1d4dc', background: { type: 'solid', color: '#131722' } }, grid: { vertLines: { color: '#2B2B43' }, horzLines: { color: '#2B2B43' } }, rightPriceScale: { autoScale: true, scaleMargins: { top: 0.1, bottom: 0.1 } } };
        const mainChart = LightweightCharts.createChart(document.getElementById('chart-container'), { ...commonOptions, timeScale: { timeVisible: true } });
        const candleSeries = mainChart.addCandlestickSeries({ upColor: '#26a69a', downColor: '#ef5350', borderVisible: false, wickUpColor: '#26a69a', wickDownColor: '#ef5350' });
        const volumeChart = LightweightCharts.createChart(document.getElementById('volume-container'), { ...commonOptions, timeScale: { timeVisible: true, visible: false } });
        const volumeSeries = volumeChart.addHistogramSeries({ priceFormat: { type: 'volume' } });
        mainChart.timeScale().subscribeVisibleLogicalRangeChange(range => { if (range) volumeChart.timeScale().setVisibleLogicalRange(range); });
        volumeChart.timeScale().subscribeVisibleLogicalRangeChange(range => { if (range) mainChart.timeScale().setVisibleLogicalRange(range); });

        const overlay = document.getElementById('ob-overlay'); const ctx = overlay.getContext('2d');
        function resizeOverlay() { overlay.width = document.getElementById('main-chart-wrapper').clientWidth; overlay.height = document.getElementById('main-chart-wrapper').clientHeight; }

        function drawArrowCanvas(ctx, fromx, fromy, tox, toy, color) {
            let headlen = 22; let dx = tox - fromx; let dy = toy - fromy; let angle = Math.atan2(dy, dx);
            ctx.beginPath(); ctx.moveTo(fromx, fromy); ctx.lineTo(tox, toy); ctx.strokeStyle = color; ctx.lineWidth = 5; ctx.stroke();
            ctx.beginPath(); ctx.moveTo(tox, toy); ctx.lineTo(tox - headlen * Math.cos(angle - Math.PI / 6), toy - headlen * Math.sin(angle - Math.PI / 6));
            ctx.lineTo(tox - headlen * Math.cos(angle + Math.PI / 6), toy - headlen * Math.sin(angle + Math.PI / 6));
            ctx.lineTo(tox, toy); ctx.fillStyle = color; ctx.fill();
        }

        function renderCanvasLoop() {
            requestAnimationFrame(renderCanvasLoop);
            ctx.clearRect(0, 0, overlay.width, overlay.height);
            
            if (opts.obValid || opts.stratOb) {
                validOBsList.forEach(ob => {
                    if (!ob.isValidated) return; 
                    let x1 = mainChart.timeScale().timeToCoordinate(ob.time); if (x1 === null) return; 
                    let x2 = ob.mitigatedTime ? mainChart.timeScale().timeToCoordinate(ob.mitigatedTime) : mainChart.timeScale().width(); if (x2 === null || x2 < x1) return; 
                    let y1 = candleSeries.priceToCoordinate(ob.high); let y2 = candleSeries.priceToCoordinate(ob.low); if (y1 === null || y2 === null) return;
                    let topY = Math.min(y1, y2); let heightY = Math.abs(y1 - y2); let widthX = Math.max(x2 - x1, 2);
                    ctx.fillStyle = ob.type === 'bull' ? "rgba(255, 152, 0, 0.2)" : "rgba(156, 39, 176, 0.2)"; ctx.strokeStyle = ob.type === 'bull' ? "rgba(255, 152, 0, 0.8)" : "rgba(156, 39, 176, 0.8)";
                    ctx.fillRect(x1, topY, widthX, heightY); ctx.lineWidth = 1; ctx.strokeRect(x1, topY, widthX, heightY);
                });
            }

            if (opts.liq || opts.stratLiq) {
                liquidityList.forEach(liq => {
                    let x1 = mainChart.timeScale().timeToCoordinate(liq.time); if (x1 === null) return; 
                    let x2 = liq.sweptTime ? mainChart.timeScale().timeToCoordinate(liq.sweptTime) : mainChart.timeScale().width(); if (x2 === null || x2 < x1) return; 
                    let y = candleSeries.priceToCoordinate(liq.price); if (y === null) return;
                    ctx.beginPath(); ctx.setLineDash([4, 4]); ctx.moveTo(x1, y); ctx.lineTo(x2, y); ctx.lineWidth = opts.stratLiq ? 2 : 1.5; ctx.strokeStyle = liq.type === 'BSL' ? "#ef5350" : "#26a69a"; ctx.stroke(); ctx.setLineDash([]); 
                    ctx.fillStyle = liq.type === 'BSL' ? "#ef5350" : "#26a69a"; ctx.font = "11px Arial"; ctx.fillText(liq.sweptTime ? "✖ Swept" : "💰 " + liq.type, x2 - 45, y - 5);
                });
            }

            if (activeSetup && opts.mentor) {
                let x = mainChart.timeScale().timeToCoordinate(activeSetup.time);
                let y = candleSeries.priceToCoordinate(activeSetup.entry);
                if (x !== null && y !== null) {
                    let pulse = activeSetup.finished ? 0 : (Math.sin(Date.now() / 150) + 1) / 2; 
                    let radius = 30 + pulse * 25; 
                    let colorRGB = activeSetup.type === 'LONG' ? '76, 175, 80' : '239, 83, 80';
                    let colorHex = activeSetup.type === 'LONG' ? '#4caf50' : '#ef5350';
                    
                    if(activeSetup.stars === 999) { colorRGB = '253, 216, 53'; colorHex = '#fdd835'; } // Builder
                    else if(activeSetup.stars === 100) { colorRGB = '0, 229, 255'; colorHex = '#00e5ff'; } // V2
                    else if(activeSetup.stars === 99) { colorRGB = '224, 64, 251'; colorHex = '#e040fb'; } // V1

                    ctx.beginPath(); ctx.arc(x, y, radius, 0, 2 * Math.PI);
                    ctx.fillStyle = "rgba(" + colorRGB + ", " + (0.15 + pulse * 0.3) + ")"; ctx.fill();
                    ctx.beginPath(); ctx.arc(x, y, 8, 0, 2 * Math.PI);
                    ctx.fillStyle = "rgba(" + colorRGB + ", 1)"; ctx.shadowBlur = activeSetup.finished ? 0 : 20; ctx.shadowColor = "rgba(" + colorRGB + ", 1)"; ctx.fill(); ctx.shadowBlur = 0; 
                    if (activeSetup.type === 'LONG') drawArrowCanvas(ctx, x, y + 90, x, y + 25, colorHex);
                    else drawArrowCanvas(ctx, x, y - 90, x, y - 25, colorHex);
                }
            }
        }
        renderCanvasLoop();

        
        let coachSlParams = { atr_multiplier: 2.0, rr_target: 2.0 };
        function updateCoachMemoryParams() {
            fetch('/api/sl_params')
            .then(res => res.json())
            .then(data => { if(data.atr_multiplier) coachSlParams = data; })
            .catch(e => console.error("Erreur de MAJ des parametres SL", e));
        }
        updateCoachMemoryParams();

        // Fonction mathématique intelligente pour le Stop Loss (respect strict de la position et de l'historique)
        function calculateSmartSLTP(type, entryPrice, atr, supportResist) {
            let mult = coachSlParams.atr_multiplier || 2.0;
            let rr = coachSlParams.rr_target || 2.0;
            let sl, tp;

            if (type === 'LONG') {
                let mathSL = entryPrice - (atr * mult);
                sl = supportResist ? Math.min(supportResist, mathSL) : mathSL;
                // Protection absolue : Le SL DOIT être en dessous de l'entrée
                if (sl >= entryPrice) sl = entryPrice - (atr * mult);
                tp = entryPrice + (Math.abs(entryPrice - sl) * rr);
            } else { // SHORT
                let mathSL = entryPrice + (atr * mult);
                sl = supportResist ? Math.max(supportResist, mathSL) : mathSL;
                // Protection absolue : Le SL DOIT être au dessus de l'entrée
                if (sl <= entryPrice) sl = entryPrice + (atr * mult);
                tp = entryPrice - (Math.abs(sl - entryPrice) * rr);
            }
            return { sl, tp };
        }

        function checkActiveTrade(currentPrice) {
            if (!activeSetup || activeSetup.finished) return;
            const statusDiv = document.getElementById('scanner-status');
            const responseDiv = document.getElementById('ai-response');
            
            let result = null;
            if (activeSetup.type === 'LONG') { if (currentPrice >= activeSetup.tp) result = 'WIN'; else if (currentPrice <= activeSetup.sl) result = 'LOSS'; } 
            else { if (currentPrice <= activeSetup.tp) result = 'WIN'; else if (currentPrice >= activeSetup.sl) result = 'LOSS'; }

            if (result) {
                activeSetup.finished = true;
                let pnl = Math.abs(activeSetup.tp - activeSetup.entry)/activeSetup.entry * 100;
                let msg = result === 'WIN' ? "🎯 CIBLE ATTEINTE (+" + pnl.toFixed(2) + "%)" : "❌ STOP LOSS TOUCHÉ";
                let color = result === 'WIN' ? '#4caf50' : '#ef5350';
                statusDiv.innerHTML = "<span style='color:" + color + "; font-weight:bold; font-size:16px;'>" + msg + "</span>";
                statusDiv.style.borderColor = color;
                
                responseDiv.innerHTML += `
                <button onclick='window.clearActiveSetup()' style='margin-top:15px; width:100%; background:#2b3139; color:white; border:1px solid ${color}; padding:10px; border-radius:4px; cursor:pointer; font-weight:bold; font-size:14px; transition:0.2s;'>
                    ${result === 'WIN' ? '✅' : '❌'} Résultat enregistré. Nettoyer et Relancer le Scanner 🔄
                </button>`;

                // NOUVEAU : Envoi des données au backend pour apprentissage
                fetch('/api/learn', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        type: activeSetup.type,
                        entry: activeSetup.entry,
                        tp: activeSetup.tp,
                        sl: activeSetup.sl,
                        reason: activeSetup.reason,
                        result: result,
                        pnl_percent: result === 'WIN' ? pnl : -pnl,
                        timestamp: new Date().toISOString()
                    })
                }).catch(err => console.error("Erreur d'enregistrement mémoire:", err));
            }
        }

        function runCoachScanner() {
            if (!opts.mentor) return;
            let data = window.globalCandles; if (!data || data.length < 50) return;
            let curr = data[data.length - 1]; let currentPrice = curr.close;

            if (activeSetup) { checkActiveTrade(currentPrice); return; }

            let newSetup = null;

            // --- 0. BUILDER MODULAIRE CUSTOM ---
            if (opts.stratBuilder) {
                let activeCount = 0; let voteLong = 0; let voteShort = 0; let reasons = [];
                let atr = calculateVolIndicators(data) || (currentPrice * 0.002);

                // 1. Algorithme (Croisement EMA)
                if (opts.subAlgo) {
                    activeCount++;
                    let ema9 = calculateEMA(data, 9); let ema21 = calculateEMA(data, 21);
                    if (ema9 > ema21) { voteLong++; reasons.push("Algo (Achat)"); } else { voteShort++; reasons.push("Algo (Vente)"); }
                }

                // 2. Moyenne Variance (Bollinger mean reversion)
                if (opts.subMeanVar) {
                    activeCount++;
                    let sum = 0; for(let i=data.length-20; i<data.length; i++) sum += data[i].close;
                    let mean = sum/20; let v = 0; for(let i=data.length-20; i<data.length; i++) v += Math.pow(data[i].close - mean, 2);
                    let stddev = Math.sqrt(v/20);
                    if (currentPrice < mean - stddev) { voteLong++; reasons.push("Variance (Survendu)"); } 
                    else if (currentPrice > mean + stddev) { voteShort++; reasons.push("Variance (Suracheté)"); }
                }

                // 3. Régression L2
                if (opts.subL2) {
                    activeCount++;
                    let rr = calculateRidgeRegression(data, 20, 1.5);
                    if (rr.projected > currentPrice) { voteLong++; reasons.push("L2 (Hausse)"); } 
                    else if (rr.projected < currentPrice) { voteShort++; reasons.push("L2 (Baisse)"); }
                }

                // 4. Optimisation (RSI Filtre)
                if (opts.subOpti) {
                    activeCount++;
                    let rsi = calculateRSI(data, 14);
                    if (rsi > 40 && rsi < 70) { voteLong++; reasons.push("Opti (Force)"); } 
                    else if (rsi < 60 && rsi > 30) { voteShort++; reasons.push("Opti (Faiblesse)"); }
                }

                // 5. Convexe (2nd dérivée)
                if (opts.subConvex) {
                    activeCount++;
                    let idx = data.length - 1;
                    let convex = data[idx].close - 2*data[idx-1].close + data[idx-2].close;
                    if (convex > 0) { voteLong++; reasons.push("Convexe (+)"); } else { voteShort++; reasons.push("Convexe (-)"); }
                }

                // 6. Régression Linéaire
                if (opts.subLr) {
                    activeCount++;
                    let lr = calculateRidgeRegression(data, 20, 0); // lambda 0 = Linéaire classique
                    if (lr.slope > 0) { voteLong++; reasons.push("Linéaire (Pente +)"); } else { voteShort++; reasons.push("Linéaire (Pente -)"); }
                }

                if (activeCount > 0) {
                    // CONDITION : Toutes les stratégies cochées doivent être d'accord (Confluence 100%)
                    if (voteLong === activeCount) {
                        newSetup = { type: 'LONG', entry: currentPrice, sl: currentPrice - (atr*2), tp: currentPrice + (atr*4), time: curr.time, reason: "🧪 Confluence Custom ("+activeCount+"/"+activeCount+") : " + reasons.join(' + '), stars: 999 };
                    } else if (voteShort === activeCount) {
                        newSetup = { type: 'SHORT', entry: currentPrice, sl: currentPrice + (atr*2), tp: currentPrice - (atr*4), time: curr.time, reason: "🧪 Confluence Custom ("+activeCount+"/"+activeCount+") : " + reasons.join(' + '), stars: 999 };
                    }
                }
            }

            // --- 1. SCANNER QUANT V2 (PRO DYNAMIQUE) ---
            if (!newSetup && opts.stratQuantV2) {
                let atr = calculateVolIndicators(data);
                quantData.poc = calculatePOC(data, 50);

                let kPrices = [];
                let p = 1.0, r = 0.01, q = 0.1, x = data[0].close;
                for(let i=0; i<data.length; i++) {
                    p = p + r; let k = p / (p + q); x = x + k * (data[i].close - x); p = (1 - k) * p; kPrices.push(x);
                }

                let dynLambda = 0.5 + ((atr / currentPrice) * 1000);
                let period = 20; let sumX=0, sumY=0, sumXY=0, sumXX=0;
                for(let i=0; i<period; i++) { let xi=i; let yi=kPrices[kPrices.length-period+i]; sumX+=xi; sumY+=yi; sumXY+=xi*yi; sumXX+=xi*xi; }
                let meanX=sumX/period, meanY=sumY/period; let num=0, den=0;
                for(let i=0; i<period; i++) { let xi=i-meanX; let yi=kPrices[kPrices.length-period+i]-meanY; num+=xi*yi; den+=xi*xi; }
                let slopeV2 = num / (den + dynLambda); let projV2 = (slopeV2 * period) + (meanY - slopeV2 * meanX);

                let idx = kPrices.length - 1; let convexity = kPrices[idx] - 2*kPrices[idx-1] + kPrices[idx-2]; quantData.v2Convexity = convexity;

                let scoreLong = 50; let scoreShort = 50;
                if(slopeV2 > 0) scoreLong += 15; else scoreShort += 15;
                if(projV2 > currentPrice) scoreLong += 10; else scoreShort += 10;
                if(currentPrice > quantData.poc) { scoreLong += 15; scoreShort -= 10; } else { scoreShort += 15; scoreLong -= 10; }
                if(currentPrice > quantData.supertrend) scoreLong += 10; else scoreShort += 10;
                if(convexity > 0) scoreLong += 15; if(convexity < 0) scoreShort += 15; 
                scoreLong = Math.min(scoreLong, 99); scoreShort = Math.min(scoreShort, 99);

                if (scoreLong >= 80 && currentPrice > quantData.poc) newSetup = { type: 'LONG', entry: currentPrice, sl: Math.min(quantData.supertrend, currentPrice - atr * 2), tp: currentPrice + (projV2 - currentPrice)*2.5, time: curr.time, reason: `Filtre Kalman purifié + Convexité Haussière. Score: ${scoreLong}%`, stars: 100 };
                else if (scoreShort >= 80 && currentPrice < quantData.poc) newSetup = { type: 'SHORT', entry: currentPrice, sl: Math.max(quantData.supertrend, currentPrice + atr * 2), tp: currentPrice - (currentPrice - projV2)*2.5, time: curr.time, reason: `Filtre Kalman purifié + Convexité Baissière. Score: ${scoreShort}%`, stars: 100 };
            }

            // --- 2. SCANNER QUANT V1 ---
            if (!newSetup && opts.stratQuant) {
                let rrObj = calculateRidgeRegression(data, 20, 1.5);
                quantData.poc = calculatePOC(data, 50); let atr = calculateVolIndicators(data);
                let diffProj = Math.abs(rrObj.projected - currentPrice) / currentPrice;
                if (diffProj > 0.0005) { 
                    if (rrObj.projected > currentPrice && currentPrice > quantData.poc && currentPrice > quantData.supertrend) newSetup = { type: 'LONG', entry: currentPrice, sl: Math.min(quantData.supertrend, currentPrice - atr * 2), tp: currentPrice + (rrObj.projected - currentPrice)*2, time: curr.time, reason: "Optimisation L2 (V1) projette un momentum positif. Soutien POC.", stars: 99 };
                    else if (rrObj.projected < currentPrice && currentPrice < quantData.poc && currentPrice < quantData.supertrend) newSetup = { type: 'SHORT', entry: currentPrice, sl: Math.max(quantData.supertrend, currentPrice + atr * 2), tp: currentPrice - (currentPrice - rrObj.projected)*2, time: curr.time, reason: "La Régression L2 (V1) indique une déviation baissière sous le POC.", stars: 99 };
                }
            }

            // --- 3. SCANNER SMC ---
            if (!newSetup && opts.stratOb && opts.stratLiq) {
                let lastValidOb = validOBsList.slice().reverse().find(ob => !ob.mitigatedTime);
                let recentSweeps = liquidityList.filter(l => l.sweptTime).slice(-2);
                let activeLiq = liquidityList.filter(l => !l.sweptTime);
                if (lastValidOb && recentSweeps.length > 0) {
                    let sweep = recentSweeps[0];
                    if ((curr.time - sweep.sweptTime) < 500 && currentPrice >= lastValidOb.low && currentPrice <= lastValidOb.high) {
                        if (sweep.type === 'BSL' && lastValidOb.type === 'bear') newSetup = { type: 'SHORT', entry: currentPrice, sl: lastValidOb.high * 1.001, tp: (activeLiq.find(l=>l.type==='SSL'&&l.price<currentPrice)||{price:currentPrice*0.98}).price, time: curr.time, reason: "Manipulation (BSL Sweep) + Rejet OB Vendeur.", stars: 3 };
                        else if (sweep.type === 'SSL' && lastValidOb.type === 'bull') newSetup = { type: 'LONG', entry: currentPrice, sl: lastValidOb.low * 0.999, tp: (activeLiq.find(l=>l.type==='BSL'&&l.price>currentPrice)||{price:currentPrice*1.02}).price, time: curr.time, reason: "Manipulation (SSL Sweep) + Rebond OB Acheteur.", stars: 3 };
                    }
                }
            }

            // --- AFFICHAGE VISUEL ---
            if (newSetup) {
                let rr = Math.abs(newSetup.tp - newSetup.entry) / Math.abs(newSetup.entry - newSetup.sl);
                if (rr >= 0.5) { // Abaissé à 0.5 pour que le Builder Custom affiche plus facilement ses découvertes
                    activeSetup = newSetup; activeSetup.finished = false;
                    
                    if (setupLines && setupLines.length > 0) setupLines.forEach(l => { try { candleSeries.removePriceLine(l); } catch(e){} });
                    setupLines = [];
                    setupLines.push(candleSeries.createPriceLine({ price: activeSetup.entry, color: '#9e9e9e', lineWidth: 2, lineStyle: 0, title: 'ENTRY' }));
                    setupLines.push(candleSeries.createPriceLine({ price: activeSetup.sl, color: '#ef5350', lineWidth: 2, lineStyle: 2, title: 'STOP' }));
                    setupLines.push(candleSeries.createPriceLine({ price: activeSetup.tp, color: '#4caf50', lineWidth: 2, lineStyle: 2, title: 'CIBLE' }));

                    let isBuilder = activeSetup.stars === 999;
                    let isQuantV2 = activeSetup.stars === 100;
                    let isQuantV1 = activeSetup.stars === 99;
                    
                    let colorV = isBuilder ? '#fdd835' : (isQuantV2 ? '#00e5ff' : (isQuantV1 ? '#e040fb' : (activeSetup.type === 'LONG' ? '#4caf50' : '#ef5350')));
                    let starsHtml = isBuilder ? "🧪 LABO MODULAIRE" : (isQuantV2 ? "🧬 QUANT V2 PRO" : (isQuantV1 ? "⚛️ QUANT V1" : "⭐".repeat(activeSetup.stars)));

                    document.getElementById('scanner-status').innerHTML = "<span style='color:" + colorV + "; text-transform:uppercase;'>🚨 SETUP DÉTECTÉ : " + starsHtml + "</span>";
                    document.getElementById('scanner-status').style.borderColor = colorV;

                    document.getElementById('ai-response').innerHTML = `
                    <div style='background:rgba(0,0,0,0.5); padding:15px; border-radius:8px; border-left: 5px solid ${colorV}; box-shadow: 0 4px 15px rgba(0,0,0,0.3);'>
                        <div style='margin-bottom:12px; font-size:18px; display:flex; justify-content:space-between;'>
                            <span>🎯 <b style='color:${colorV};'>ENTRER ${activeSetup.type}</b></span>
                            <span style='font-size:12px; border:1px solid ${colorV}; padding:2px 5px; border-radius:4px; color:${colorV};'>${starsHtml}</span>
                        </div>
                        <div style='display:grid; grid-template-columns: 1fr 1fr; gap:12px; font-size:14px; margin-bottom:12px; background:#131722; padding:10px; border-radius:6px;'>
                            <div>🟢 <b>Prix:</b> ${activeSetup.entry.toFixed(4)}</div>
                            <div style='color:#fcd535;'>⚖️ <b>RR:</b> 1 : ${rr.toFixed(1)}</div>
                            <div style='color:#ef5350;'>🔴 <b>SL:</b> ${activeSetup.sl.toFixed(4)}</div>
                            <div style='color:#4caf50;'>💰 <b>TP:</b> ${activeSetup.tp.toFixed(4)}</div>
                        </div>
                        <div style='padding-top:10px; border-top:1px solid #4a5056; font-style:italic; color:#b0bec5; font-size:13px; line-height:1.5;'>
                            <b>Logique :</b> ${activeSetup.reason}
                        </div>
                        <button onclick='window.clearActiveSetup()' style='margin-top:15px; width:100%; background:#ef5350; color:white; border:none; padding:8px; border-radius:4px; cursor:pointer; font-weight:bold;'>✖ Fermer cette analyse et relancer le scan</button>
                    </div>`;
                }
            }
        }

        function calculateIndicators() {
            const data = window.globalCandles; if(!data || data.length < 15) return;
            validOBsList = []; liquidityList = []; fvgList = []; let markerMap = new Map();
            function addMarker(t, p, c, s, txt) { if(markerMap.has(t)) markerMap.get(t).text += ' | ' + txt; else markerMap.set(t, { time: t, position: p, color: c, shape: s, text: txt }); }

            const PIVOT_LENGTH = 10;
            for (let i = PIVOT_LENGTH; i < data.length - PIVOT_LENGTH; i++) {
                let isSwingHigh = true; let isSwingLow = true; let pivotPriceH = data[i].high; let pivotPriceL = data[i].low;
                for (let j = i - PIVOT_LENGTH; j <= i + PIVOT_LENGTH; j++) {
                    if (i !== j) { if (data[j].high > pivotPriceH) isSwingHigh = false; if (data[j].low < pivotPriceL) isSwingLow = false; }
                }
                if (isSwingHigh) { let liq = { type: 'BSL', time: data[i].time, price: pivotPriceH, sweptTime: null }; for(let k = i+1; k < data.length; k++) { if(data[k].high > liq.price) { liq.sweptTime = data[k].time; break; } } liquidityList.push(liq); }
                if (isSwingLow) { let liq = { type: 'SSL', time: data[i].time, price: pivotPriceL, sweptTime: null }; for(let k = i+1; k < data.length; k++) { if(data[k].low < liq.price) { liq.sweptTime = data[k].time; break; } } liquidityList.push(liq); }
            }

            for (let i = 2; i < data.length - 2; i++) {
                let curr = data[i], prev2 = data[i-2]; let isBullFVG = curr.low > prev2.high; let isBearFVG = curr.high < prev2.low;
                if (isBullFVG) { if(opts.fvg) addMarker(data[i-1].time, 'belowBar', '#26a69a', 'arrowUp', 'FVG'); fvgList.push({ type: 'bull', time: curr.time, top: curr.low, bottom: prev2.high, index: i }); }
                if (isBearFVG) { if(opts.fvg) addMarker(data[i-1].time, 'aboveBar', '#ef5350', 'arrowDown', 'FVG'); fvgList.push({ type: 'bear', time: curr.time, top: prev2.low, bottom: curr.high, index: i }); }
            }

            for (let i = 6; i < data.length - 2; i++) {
                let curr = data[i]; let lowestLow = Math.min(...data.slice(i-6, i).map(d => d.low)); let highestHigh = Math.max(...data.slice(i-6, i).map(d => d.high));
                let isSMC_BullOB = (curr.close < curr.open) && (curr.low <= lowestLow) && (data[i+2] && data[i+2].low > curr.high);
                if (isSMC_BullOB) { let ob = { type: 'bull', time: curr.time, high: curr.high, low: curr.low, mitigatedTime: null, isValidated: true }; for (let j = i+1; j < data.length; j++) { if (data[j].close < ob.low) { ob.mitigatedTime = data[j].time; break; } } validOBsList.push(ob); }
                let isSMC_BearOB = (curr.close > curr.open) && (curr.high >= highestHigh) && (data[i+2] && data[i+2].high < curr.low);
                if (isSMC_BearOB) { let ob = { type: 'bear', time: curr.time, high: curr.high, low: curr.low, mitigatedTime: null, isValidated: true }; for (let j = i+1; j < data.length; j++) { if (data[j].close > ob.high) { ob.mitigatedTime = data[j].time; break; } } validOBsList.push(ob); }
            }
            candleSeries.setMarkers(Array.from(markerMap.values()).sort((a, b) => a.time - b.time));
            runCoachScanner();
        }

        let currentSocket = null, lastCandleTime = 0;

        async function fetchHistory(symbol, interval) {
            try {
                window.clearActiveSetup();
                document.getElementById('live-price').innerText = "Téléchargement...";
                const response = await fetch("https://data-api.binance.vision/api/v3/klines?symbol=" + symbol + "&interval=" + interval + "&limit=1000");
                if (!response.ok) throw new Error("HTTP " + response.status);
                const data = await response.json();
                
                window.globalCandles = []; const volumesData = [];
                data.forEach(d => {
                    const time = Math.floor(d[0] / 1000), open = parseFloat(d[1]), close = parseFloat(d[4]);
                    window.globalCandles.push({ time, open, high: parseFloat(d[2]), low: parseFloat(d[3]), close, volume: parseFloat(d[5]) });
                    volumesData.push({ time, value: parseFloat(d[5]), color: close >= open ? '#26a69a' : '#ef5350' });
                });
                
                candleSeries.setData(window.globalCandles); volumeSeries.setData(volumesData); 
                mainChart.priceScale('right').applyOptions({ autoScale: true });
                let totalCandles = window.globalCandles.length;
                if (totalCandles > 150) mainChart.timeScale().setVisibleLogicalRange({ from: totalCandles - 150, to: totalCandles - 1 });
                else mainChart.timeScale().fitContent();

                lastCandleTime = window.globalCandles[window.globalCandles.length - 1].time;
                updatePriceDisplay(window.globalCandles[window.globalCandles.length - 1].close);
                calculateIndicators();
            } catch (error) { document.getElementById('live-price').innerText = "Erreur Connexion"; document.getElementById('live-price').style.color = "#ef5350"; }
        }

        function connectWebSocket(symbol, interval) {
            if (currentSocket) currentSocket.close();
            currentSocket = new WebSocket("wss://stream.binance.com:9443/ws/" + symbol.toLowerCase() + "@kline_" + interval);
            currentSocket.onmessage = (event) => {
                if(!window.globalCandles || window.globalCandles.length === 0) return; 
                const k = JSON.parse(event.data).k;
                const candleTime = Math.floor(k.t / 1000), closePrice = parseFloat(k.c), openPrice = parseFloat(k.o), vol = parseFloat(k.v);
                updatePriceDisplay(closePrice);
                const newCandle = { time: candleTime, open: openPrice, high: parseFloat(k.h), low: parseFloat(k.l), close: closePrice, volume: vol };
                if (candleTime >= lastCandleTime) {
                    candleSeries.update(newCandle); volumeSeries.update({ time: candleTime, value: vol, color: closePrice >= openPrice ? '#26a69a' : '#ef5350' });
                    if (candleTime > lastCandleTime) window.globalCandles.push(newCandle); else window.globalCandles[window.globalCandles.length - 1] = newCandle;
                    lastCandleTime = candleTime; calculateIndicators(); 
                }
            };
        }

        function loadChart(symbol, interval) { fetchHistory(symbol, interval).then(() => connectWebSocket(symbol, interval)); }

        document.querySelectorAll('#timeframes button[data-interval]').forEach(btn => {
            btn.addEventListener('click', (e) => { document.querySelectorAll('#timeframes button').forEach(b => b.classList.remove('active')); e.target.classList.add('active'); currentInterval = e.target.dataset.interval; loadChart(currentSymbol, currentInterval); });
        });
        document.getElementById('symbol-select').addEventListener('change', (e) => { currentSymbol = e.target.value; loadChart(currentSymbol, currentInterval); });
        document.getElementById('toggle-liq').addEventListener('change', (e) => { opts.liq = e.target.checked; calculateIndicators(); });
        document.getElementById('toggle-fvg').addEventListener('change', (e) => { opts.fvg = e.target.checked; calculateIndicators(); });
        document.getElementById('toggle-ob-valid').addEventListener('change', (e) => { opts.obValid = e.target.checked; calculateIndicators(); });
        document.getElementById('toggle-mentor').addEventListener('change', (e) => { opts.mentor = e.target.checked; coachHud.style.display = opts.mentor ? 'block' : 'none'; if(!opts.mentor) window.clearActiveSetup(); });

        document.getElementById('toggle-strat-ob').addEventListener('change', (e) => { opts.stratOb = e.target.checked; updateTags(); calculateIndicators(); });
        document.getElementById('toggle-strat-liq').addEventListener('change', (e) => { opts.stratLiq = e.target.checked; updateTags(); calculateIndicators(); });
        document.getElementById('toggle-strat-quant').addEventListener('change', (e) => { opts.stratQuant = e.target.checked; updateTags(); calculateIndicators(); });
        document.getElementById('toggle-strat-quant-v2').addEventListener('change', (e) => { opts.stratQuantV2 = e.target.checked; updateTags(); calculateIndicators(); });

        // Builder Events
        document.getElementById('toggle-strat-builder').addEventListener('change', (e) => { 
            opts.stratBuilder = e.target.checked; 
            document.getElementById('builder-options').style.display = opts.stratBuilder ? 'block' : 'none';
            updateTags(); calculateIndicators(); 
        });
        document.querySelectorAll('.sub-strat').forEach(chk => {
            chk.addEventListener('change', (e) => {
                let id = e.target.id;
                if(id === 'sub-algo') opts.subAlgo = e.target.checked;
                if(id === 'sub-meanvar') opts.subMeanVar = e.target.checked;
                if(id === 'sub-l2') opts.subL2 = e.target.checked;
                if(id === 'sub-opti') opts.subOpti = e.target.checked;
                if(id === 'sub-convex') opts.subConvex = e.target.checked;
                if(id === 'sub-lr') opts.subLr = e.target.checked;
                calculateIndicators();
            });
        });

        const resizer = document.getElementById('resizer'); const volumeContainer = document.getElementById('volume-container'); let isResizing = false;
        resizer.addEventListener('mousedown', () => { isResizing = true; document.body.style.cursor = 'row-resize'; });
        window.addEventListener('mousemove', (e) => {
            if (!isResizing) return;
            const newHeight = window.innerHeight - e.clientY - 40; 
            if (newHeight > 50 && newHeight < 400) { volumeContainer.style.height = newHeight + "px"; mainChart.resize(document.getElementById('main-chart-wrapper').clientWidth, document.getElementById('main-chart-wrapper').clientHeight); volumeChart.resize(volumeContainer.clientWidth, volumeContainer.clientHeight); resizeOverlay(); }
        });
        window.addEventListener('mouseup', () => { isResizing = false; document.body.style.cursor = 'default'; });
        new ResizeObserver(entries => { if (entries.length > 0) { const wrapper = document.getElementById('main-chart-wrapper'); mainChart.applyOptions({ width: wrapper.clientWidth }); volumeChart.applyOptions({ width: volumeContainer.clientWidth }); resizeOverlay(); } }).observe(document.querySelector('.charts-wrapper'));

        resizeOverlay(); updateTags(); loadChart(currentSymbol, currentInterval); 
    </script>
</body>
</html>"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    app.run(debug=True)
