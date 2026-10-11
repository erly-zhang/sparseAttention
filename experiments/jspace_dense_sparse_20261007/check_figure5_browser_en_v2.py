"""Exercise the local file:// package without any network requests."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
HERE=Path(__file__).resolve().parent
OUT=HERE/'figure5_html_en_v2'

def ready(page):
    page.wait_for_function("document.getElementById('status').textContent.includes('Ranks ready')",timeout=60000)

def main():
    errors=[];remote=[];checks=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1840,'height':1300},device_scale_factor=1)
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.on('request',lambda request:remote.append(request.url) if request.url.startswith(('http:','https:')) else None)
        page.goto((OUT/'index.html').as_uri());ready(page)
        assert page.locator('.readrow').count()==88
        assert page.locator('.displaynotice').is_visible()
        assert page.title().endswith('v2')
        checks.append('file:// startup, 52 reasoning positions and 36 layers')
        page.screenshot(path=str(OUT/'preview.png'),full_page=True)
        page.locator('#arm').select_option('full_w1');ready(page)
        assert page.evaluate('FIGURE5_STATE.arm')==2
        before=page.evaluate('FIGURE5_STATE.data.labels[FIGURE5_STATE.point].step')
        page.locator('#next').click();ready(page)
        assert page.evaluate('FIGURE5_STATE.data.labels[FIGURE5_STATE.point].step')>before
        page.locator('#layer').select_option('35');ready(page)
        assert 'model output' in page.locator('#positionSubtitle').inner_text()
        checks.append('method, checkpoint and layer selection')
        page.locator('#reasoningOnly').uncheck();ready(page)
        page.locator('#position').select_option('0');ready(page)
        assert 'prefill' in page.locator('#positionInfo').inner_text()
        checks.append('prefill and non-reasoning inclusion')
        page.locator('#clearPins').click();ready(page)
        assert page.evaluate('FIGURE5_STATE.pins.length')==0
        page.locator('#cellReadout button').first.click();ready(page)
        assert page.evaluate('FIGURE5_STATE.pins.length')==1
        page.locator('#search').fill('sum')
        candidate=page.evaluate("Array.from(document.getElementById('searchResults').options).find(o=>!FIGURE5_STATE.pins.includes(Number(o.value))).value")
        page.locator('#searchResults').select_option(candidate)
        page.locator('#addPin').click();ready(page)
        assert page.evaluate('FIGURE5_STATE.pins.length')==2
        page.locator('#baseline').uncheck();ready(page)
        assert page.locator('#positionChart polyline').count()==2
        checks.append('pin/unpin, search and Dense overlay toggle')
        for idx in (0,1,2):
            page.locator('#sample').select_option(str(idx));ready(page)
            for arm in ['dense','full_random','full_w1','compact_random','compact_w1']:
                page.locator('#arm').select_option(arm);ready(page)
                assert page.locator('#cellReadout tbody tr').count()==8
                english_audit=page.evaluate('''() => {
                  const d=FIGURE5_STATE.data;
                  const valid=Object.values(d.vocab).every(t=>/^[\\x00-\\x7F]*$/.test(t)&&/[A-Za-z0-9]/.test(t));
                  const uiHasHan=/[\\u3400-\\u9fff]/.test(document.body.innerText);
                  return {valid,uiHasHan,filter:d.display_filter};
                }''')
                assert english_audit=={'valid':True,'uiHasHan':False,'filter':'english'},english_audit
            audit=page.evaluate('''async () => {
              const s=FIGURE5_STATE,d=s.data;
              let checked=0;
              for(const tid of s.pins){
                const r=await ranks(tid);
                for(let a=0;a<5;a++)for(let p=0;p<d.P;p++)for(let l=0;l<36;l++)for(let k=0;k<d.N;k++){
                  const i=((a*d.P+p)*36+l)*d.N+k;
                  if(d.top_ids[i]!==tid)continue;
                  const j=((a*d.P+p)*36+l)*2;
                  if(r[j]!==d.top_low[i]||r[j+1]!==d.top_high[i])throw Error('Top8/rank mismatch');checked++;
                }
              }
              return {sample:d.idx,checked,tracked:d.tracked.length};
            }''')
            assert audit['checked']>0;checks.append(audit)
        page.locator('#sample').select_option('0');ready(page)
        page.locator('#reasoningOnly').check();ready(page)
        page.locator('#layer').select_option('17');ready(page)
        page.locator('#position').select_option(str(page.evaluate('FIGURE5_STATE.data.labels.findIndex(r=>r.step===2560)')));ready(page)
        page.locator('#baseline').check();ready(page)
        # Select an actual matrix cell, independent of dropdown controls.
        page.locator('#topScroll').evaluate('(el)=>el.scrollLeft=0')
        page.locator('#topGrid').click(position={'x':40,'y':37+18*5+8})
        ready(page);assert page.evaluate('FIGURE5_STATE.layer')==30
        checks.append('grid click selects actual captured coordinate')
        page.locator('#layer').select_option('17');ready(page)
        page.locator('#position').select_option(str(page.evaluate('FIGURE5_STATE.data.labels.findIndex(r=>r.step===2560)')));ready(page)
        page.evaluate('scrollToPoint()')
        page.mouse.move(10,10)
        page.screenshot(path=str(OUT/'preview.png'),full_page=True)
        page.set_viewport_size({'width':390,'height':844});ready(page)
        assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
        page.screenshot(path=str(HERE/'figure5_mobile_check_en_v2.png'),full_page=True)
        checks.append('390px viewport: no page horizontal overflow')
        browser.close()
    result=dict(status='passed' if not errors and not remote else 'failed',checks=checks,page_errors=errors,external_requests=remote)
    (OUT/'browser_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))
    assert not errors and not remote

if __name__=='__main__':main()
