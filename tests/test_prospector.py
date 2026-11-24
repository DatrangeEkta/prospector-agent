"""
Automated checks for the Technique Prospector. Run before every change:

    python3 -m unittest discover -s tests -v

Everything runs offline in a temporary folder: the saved arXiv sample feed instead of the internet, a fake
embedding model instead of Ollama, and no AI notes. The real settings, scans, pilots and radar page are never
touched. The page test renders radar.html in headless Chrome and is skipped when Chrome is not installed.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import zlib

import numpy as np
import openpyxl
import pandas as pd

AGENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(AGENT, "src")
sys.path.insert(0, SRC)
import prospector as P  # noqa: E402
import build_radar as B  # noqa: E402

FEED = os.path.join(AGENT, "tests", "fixtures", "arxiv_feed_sample.xml")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def fake_embed(texts, timeout=300):
    """Bag of words hashed into 64 numbers: texts sharing words point the same way, like a real embedding."""
    out = []
    for t in texts:
        v = np.full(64, 1e-3)
        for w in re.findall(r"[a-z]+", t.lower()):
            v[zlib.crc32(w.encode()) % 64] += 1
        out.append(v.tolist())
    return out


def read(path):
    with open(path) as f:
        return f.read()


def run_of(out):
    return json.loads(read(out + ".json"))["run"]


def scan_args(out, **kw):
    a = dict(feed=FEED, days=30, month=None, top=5, include_seen=False, no_model=True, no_meaning=False, out=out)
    a.update(kw)
    return argparse.Namespace(**a)


class Sandbox(unittest.TestCase):
    """Points every file the agent reads or writes at a temporary folder."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmp, "scans"))
        self.saved = {k: getattr(P, k) for k in ("HERE", "SEEN", "PILOTS", "CONFIG", "SETTINGS", "EMB_CACHE",
                                                   "LABELS", "VOTES", "RUN_HISTORY", "MIN_PAPERS", "refresh_radar",
                                                   "POOL", "ASK_LOG")}
        self.saved_embed, self.saved_b = P.models.embed, (B.HERE, B.OUT)
        P.HERE = self.tmp
        P.SEEN, P.PILOTS = (os.path.join(self.tmp, f) for f in ("papers_seen.csv", "pilots.csv"))
        P.CONFIG, P.SETTINGS = (os.path.join(self.tmp, f) for f in ("interests.json", "interests.xlsx"))
        P.EMB_CACHE, P.LABELS, P.VOTES = (os.path.join(self.tmp, f) for f in
                                          ("emb.npz", "eval_labels.xlsx", "votes.xlsx"))
        P.RUN_HISTORY = os.path.join(self.tmp, "scans", "run_history.csv")
        P.POOL, P.ASK_LOG = os.path.join(self.tmp, "pool.jsonl"), os.path.join(self.tmp, "scans", "ask_log.csv")
        P.refresh_radar = lambda *a, **k: None   # pick/result must not rebuild the real page
        P.models.embed = fake_embed
        B.HERE, B.OUT = self.tmp, os.path.join(self.tmp, "radar.html")

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(P, k, v)
        P.models.embed = self.saved_embed
        B.HERE, B.OUT = self.saved_b
        shutil.rmtree(self.tmp, ignore_errors=True)

    def scan(self, name="Prospector_test", **kw):
        out = os.path.join(self.tmp, "scans", name)
        self.assertEqual(P.cmd_scan(scan_args(out, **kw)), 0)
        return out

    def settings_file(self):
        P.write_settings(P.DEFAULT_CONFIG, P.SETTINGS)
        return P.SETTINGS


class TestSettings(Sandbox):
    def test_excel_round_trip_keeps_every_setting(self):
        cfg = P.read_settings(self.settings_file(), dict(P.DEFAULT_CONFIG))
        self.assertEqual(cfg["interests"], P.DEFAULT_CONFIG["interests"])
        self.assertEqual(cfg["exclude"], P.DEFAULT_CONFIG["exclude"])
        self.assertEqual(len(cfg["work"]), len(P.DEFAULT_CONFIG["work"]))
        self.assertEqual(cfg["meaning_weight"], 50)

    def test_mistakes_are_reported_with_the_row(self):
        wb = openpyxl.load_workbook(self.settings_file())
        wb["Topics"]["B2"] = "Very high"
        wb["Topics"].append(["agent", "High", "No", ""])
        wb.save(P.SETTINGS)
        with self.assertRaises(SystemExit) as e:
            P.read_settings(P.SETTINGS, dict(P.DEFAULT_CONFIG))
        self.assertIn("row 2: importance 'Very high'", str(e.exception))
        self.assertIn("'agent' is listed twice", str(e.exception))

    def test_no_topic_for_ai_papers_is_refused(self):
        wb = openpyxl.load_workbook(self.settings_file())
        for row in wb["Topics"].iter_rows(min_row=2):
            row[1].value = row[1].value or "Low"   # find-only topics need an importance once they stop finding
            row[2].value = "No"
        wb.save(P.SETTINGS)
        with self.assertRaises(SystemExit) as e:
            P.read_settings(P.SETTINGS, dict(P.DEFAULT_CONFIG))
        self.assertIn("no LLM / AI papers would be downloaded", str(e.exception))


class TestMyTasks(Sandbox):
    def add_tasks(self, rows):
        wb = openpyxl.load_workbook(self.settings_file())
        for r in rows:
            wb["My tasks"].append(r)
        wb.save(P.SETTINGS)
        return P.read_settings(P.SETTINGS, dict(P.DEFAULT_CONFIG))

    def test_recent_tasks_are_used_and_old_ones_are_not(self):
        today = P.datetime.now().strftime("%Y-%m-%d")
        cfg = self.add_tasks([[today, "Pulling fee tables out of scanned lease agreements"],
                              ["2020-01-15", "An old task from years ago"]])
        self.assertEqual(len(cfg["tasks"]), 1)
        self.assertIn("lease agreements", cfg["tasks"][0]["text"])

    def test_a_task_without_a_date_is_reported(self):
        with self.assertRaises(SystemExit) as e:
            self.add_tasks([["", "A task with no date"]])
        self.assertIn("sheet My tasks, row 2", str(e.exception))

    def test_a_task_can_be_the_closest_match(self):
        task = {"name": "My task: lease fee tables", "text": "pulling fee tables from scanned lease agreements",
                "group": "", "task": True}
        paper = {"arXiv ID": "2610.99999", "Title": "Fee tables from scanned lease agreements",
                 "Abstract": "pulling tables from scanned lease agreements", "Published": "2026-10-30"}
        m = P.meaning([paper], P.DEFAULT_CONFIG["work"] + [task])
        self.assertEqual(m["2610.99999"][1], "My task: lease fee tables")


class TestRanking(Sandbox):
    def paper(self, title, abstract="", published="2026-10-30"):
        return {"arXiv ID": title[:8], "Title": title, "Abstract": abstract, "Published": published}

    def test_keyword_in_title_scores_and_plain_paper_does_not(self):
        newest = P.datetime(2026, 10, 30)
        hit, _ = P.score(self.paper("Reconciliation of fund data"), P.DEFAULT_CONFIG, newest)
        miss, _ = P.score(self.paper("Bird migration patterns"), P.DEFAULT_CONFIG, newest)
        self.assertGreater(hit, 0)
        self.assertEqual(miss, 0)

    def test_skip_word_removes_a_paper(self):
        p = self.paper("Reconciliation on the blockchain")
        self.assertTrue(P.excluded(p, P.DEFAULT_CONFIG))

    def test_meaning_finds_the_closest_work(self):
        papers = [self.paper("Extracting dates fees names from long contracts with OCR", "pdf documents records"),
                  self.paper("Backtesting factor strategies", "overfitting look-ahead bias trading")]
        m = P.meaning(papers, P.DEFAULT_CONFIG["work"])
        self.assertEqual(m[papers[0]["arXiv ID"]][1], "Contract data recon")
        self.assertEqual(m[papers[1]["arXiv ID"]][1], "Quant research")

    def test_no_embedding_model_means_keywords_only(self):
        P.models.embed = lambda texts, timeout=300: None
        self.assertIsNone(P.meaning([self.paper("Anything")], P.DEFAULT_CONFIG["work"]))


class TestScan(Sandbox):
    def test_scan_writes_shortlist_scores_and_run_report(self):
        out = self.scan()
        sheets = pd.read_excel(out + ".xlsx", sheet_name=None)
        self.assertEqual(len(sheets["Shortlist"]), 5)
        self.assertIn("All Scored", sheets)
        self.assertTrue(sheets["Shortlist"]["Meaning"].notna().all())
        run = run_of(out)
        self.assertEqual(run["meaning"], "on")
        self.assertIsInstance(run["warnings"], list)

    def test_second_run_in_the_same_month_keeps_the_shortlist(self):
        """The blank-page bug: a re-run must not drop this month's papers."""
        first = pd.read_excel(self.scan() + ".xlsx", sheet_name="Shortlist")
        second = pd.read_excel(self.scan() + ".xlsx", sheet_name="Shortlist")
        self.assertEqual(len(second), len(first))
        self.assertEqual(set(second["arXiv ID"].astype(str)), set(first["arXiv ID"].astype(str)))

    def test_papers_from_an_earlier_month_are_skipped(self):
        out = self.scan()
        seen = pd.read_csv(P.SEEN, dtype=str)
        seen["First seen"], seen["Month"] = "2000-01-01", "2000-01"
        seen.to_csv(P.SEEN, index=False)
        out = self.scan()
        self.assertEqual(len(pd.read_excel(out + ".xlsx", sheet_name="Shortlist")), 0)
        warnings = run_of(out)["warnings"]
        self.assertTrue(any("no paper matched" in w for w in warnings))

    def test_too_few_papers_gives_a_warning(self):
        P.MIN_PAPERS = 10_000
        out = self.scan()
        warnings = run_of(out)["warnings"]
        self.assertTrue(any("only 150 papers downloaded" in w for w in warnings))

    def test_meaning_off_falls_back_with_a_warning(self):
        P.models.embed = lambda texts, timeout=300: None
        out = self.scan()
        run = run_of(out)
        self.assertEqual(run["meaning"], "off")
        self.assertTrue(any("Meaning ranking was off" in w for w in run["warnings"]))


class TestMonth(Sandbox):
    def test_month_range_covers_the_whole_calendar_month(self):
        start, end = P.month_range("2026-10")
        self.assertEqual((start.day, end.day, end.month), (1, 31, 10))
        self.assertEqual(P.month_range("2026-02")[1].day, 28)

    def test_month_scan_is_saved_under_that_month(self):
        out = self.scan("Prospector_2026-10", month="2026-10")
        self.assertEqual(json.loads(read(out + ".json"))["month"], "2026-10")


class TestKeepPage(Sandbox):
    def test_empty_shortlist_leaves_the_page_alone(self):
        called = []
        P.refresh_radar = lambda *a, **k: called.append(1)
        self.scan()                       # first run: papers shown, page rebuilt only when not --out
        seen = pd.read_csv(P.SEEN, dtype=str)
        seen["First seen"], seen["Month"] = "2000-01-01", "2000-01"
        seen.to_csv(P.SEEN, index=False)
        P.cmd_scan(scan_args(None))       # no --out, every paper from an earlier month: empty shortlist
        self.assertEqual(called, [])


class TestShareFolder(Sandbox):
    def test_page_is_copied_under_a_fixed_name(self):
        share = os.path.join(self.tmp, "OneDrive", "AI Research")
        saved = P.load_config
        P.load_config = lambda: {**saved(), "share_folder": share}
        try:
            page = os.path.join(self.tmp, "radar.html")
            with open(page, "w") as f:
                f.write("<p>November</p>")
            P.share_page(page)
            with open(page, "w") as f:
                f.write("<p>December</p>")
            dest = P.share_page(page)
        finally:
            P.load_config = saved
        self.assertEqual(os.path.basename(dest), P.SHARED_NAME)
        self.assertEqual(read(dest), "<p>December</p>")   # same file, latest content
        self.assertEqual(os.listdir(share), [P.SHARED_NAME])


class TestWebApp(Sandbox):
    def setUp(self):
        super().setUp()
        sys.path.insert(0, SRC)
        import app as webapp
        from fastapi.testclient import TestClient
        self.web, self.saved_home = webapp, webapp.HOME
        webapp.HOME = self.tmp
        os.makedirs(os.path.join(self.tmp, "radar"), exist_ok=True)
        self.month = "2026-10"
        out = self.scan(f"Prospector_{self.month}", month=self.month)
        B.build(out + ".xlsx", out=os.path.join(self.tmp, "scans", f"Prospector_{self.month}_radar.html"))
        self.ids = pd.read_excel(out + ".xlsx", sheet_name="Shortlist", dtype={"arXiv ID": str})["arXiv ID"].tolist()
        self.client = TestClient(webapp.app)

    def tearDown(self):
        self.web.HOME = self.saved_home
        super().tearDown()

    def test_home_serves_the_latest_page_with_the_api_hook(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn('window.PROSPECTOR_API="/api"', r.text)
        self.assertIn(f'window.PROSPECTOR_MONTH="{self.month}"', r.text)
        self.assertEqual(self.client.get("/api/months").json()["months"], [self.month])

    def test_a_vote_is_saved_and_counted(self):
        r = self.client.post("/api/votes", json={"month": self.month, "voter": "Ekta", "papers": self.ids[:2]})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.client.get(f"/api/votes?month={self.month}").json()["votes"], {"Ekta": sorted(self.ids[:2])})
        q = P.quality()
        self.assertEqual(q["months"][0]["Cards with votes"], 2)

    def test_papers_not_on_the_shortlist_are_refused(self):
        r = self.client.post("/api/votes", json={"month": self.month, "voter": "Ekta", "papers": ["9999.99999"]})
        self.assertEqual(r.status_code, 400)

    def test_tell_prospector_preview_then_add(self):
        self.settings_file()
        saved = P.models.ask_json
        P.models.ask_json = lambda *a, **k: {"topics": [{"topic": "invoice matching", "importance": "High"}],
                                             "work": [], "tasks": ["Stuck on merged cells in fee tables"], "skip": []}
        try:
            pv = self.client.post("/api/add/preview", json={"text": "my interest is invoice matching"}).json()
        finally:
            P.models.ask_json = saved
        self.assertIn("invoice matching", pv["summary"])
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "settings_backups")))   # preview writes nothing
        rows = pv["rows"]
        rows["topics"][0]["importance"] = "Super"            # a tampered row is cleaned, not trusted
        started, saved_start = [], self.web.start_scan
        self.web.start_scan = lambda: started.append(1) or True
        try:
            r = self.client.post("/api/add", json={"rows": rows})
            again = self.client.post("/api/add", json={"rows": {"topics": rows["topics"], "work": [], "tasks": [], "skip": []}})
        finally:
            self.web.start_scan = saved_start
        self.assertEqual(r.json()["added"], 2)
        self.assertTrue(r.json()["searching"])
        self.assertEqual(again.json()["added"], 0)            # already there: no new search
        self.assertEqual(len(started), 1)
        cfg = P.load_config()
        self.assertEqual(cfg["interests"]["invoice matching"], 2)   # 'Super' -> Medium
        self.assertEqual(len(cfg["tasks"]), 1)

    def test_add_with_nothing_is_refused(self):
        self.settings_file()
        r = self.client.post("/api/add", json={"rows": {"topics": [], "work": [], "tasks": [], "skip": []}})
        self.assertEqual(r.status_code, 400)

    def edited_copy(self, edit):
        r = self.client.get("/api/settings/download")
        self.assertTrue(r.content.startswith(b"PK"))
        path = os.path.join(self.tmp, "copy.xlsx")
        with open(path, "wb") as f:
            f.write(r.content)
        wb = openpyxl.load_workbook(path)
        edit(wb)
        wb.save(path)
        with open(path, "rb") as f:
            return f.read()

    def test_excel_upload_adds_new_rows_and_never_removes(self):
        self.settings_file()
        today = P.datetime.now().strftime("%Y-%m-%d")

        def edit(wb):
            ws = wb["Topics"]
            for row in list(ws.iter_rows(min_row=2)):
                if row[0].value == "agent":
                    ws.delete_rows(row[0].row)          # removed in the copy
                    break
            for row in ws.iter_rows(min_row=2):
                if row[0].value == "reconciliation":
                    row[1].value = "Low"                # changed in the copy
            ws.append(["invoice matching", "High", "Yes", ""])
            wb["My tasks"].append([today, "Matching supplier invoices against purchase orders"])
            wb["Skip"].append(["robotics"])

        body = self.edited_copy(edit)
        started, saved = [], self.web.start_scan
        self.web.start_scan = lambda: started.append(1) or True
        try:
            r = self.client.post("/api/settings/upload", content=body)
            again = self.client.post("/api/settings/upload", content=body)
        finally:
            self.web.start_scan = saved
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["new"], {"topics": 1, "work": 0, "tasks": 1, "skip": 1})
        cfg = P.load_config()
        self.assertIn("agent", cfg["interests"])              # deleted in the copy, kept in the real file
        self.assertEqual(cfg["interests"]["reconciliation"], 3)   # changed in the copy, unchanged here
        self.assertEqual(cfg["interests"]["invoice matching"], 3)
        self.assertIn("robotics", cfg["exclude"])
        self.assertEqual(again.json()["added"], 0)             # same file again: nothing new, no search
        self.assertEqual(len(started), 1)

    def test_excel_upload_with_a_mistake_is_refused(self):
        self.settings_file()
        body = self.edited_copy(lambda wb: wb["Topics"].append(["new topic", "Very high", "No", ""]))
        r = self.client.post("/api/settings/upload", content=body)
        self.assertEqual(r.status_code, 400)
        self.assertIn("should be High, Medium or Low", r.json()["detail"])
        self.assertEqual(self.client.post("/api/settings/upload", content=b"not excel").status_code, 400)

    def test_missing_month_and_scan_without_token(self):
        self.assertEqual(self.client.get("/month/2001-01").status_code, 404)
        self.assertEqual(self.client.post("/api/run").status_code, 403)


class TestAddFromMessage(Sandbox):
    def fake_model(self, reply):
        self.saved_ask = P.models.ask_json
        P.models.ask_json = lambda *a, **k: reply

    def tearDown(self):
        if hasattr(self, "saved_ask"):
            P.models.ask_json = self.saved_ask
        super().tearDown()

    def test_message_becomes_rows_and_existing_rows_are_kept(self):
        self.settings_file()
        before = pd.read_excel(P.SETTINGS, sheet_name="Topics")
        self.fake_model({"topics": [{"topic": "Invoice Matching", "importance": "very high", "find_ai_papers": True},
                                    {"topic": "agent", "importance": "High"}],
                         "work": [], "tasks": ["OCR breaks merged cells in fee tables"], "skip": ["robotics"]})
        rows = P.understand("anything", P.load_config())
        self.assertEqual(rows["topics"][0], {"topic": "invoice matching", "importance": "Medium", "find_ai": "Yes",
                                             "group": ""})   # unknown importance -> Medium
        added = P.add_to_settings(rows, P.SETTINGS)
        self.assertEqual(added["already there"], ["agent"])
        after = pd.read_excel(P.SETTINGS, sheet_name="Topics")
        self.assertEqual(len(after), len(before) + 1)
        cfg = P.load_config()   # the file still passes every check
        self.assertIn("invoice matching", cfg["interests"])
        self.assertIn("robotics", cfg["exclude"])
        self.assertEqual(len(cfg["tasks"]), 1)

    def test_a_work_area_that_only_repeats_an_interest_is_dropped(self):
        self.fake_model({"topics": [{"topic": "invoice matching", "importance": "High"}],
                         "work": [{"name": "invoice matching", "text": "My interest is invoice matching."},
                                  {"name": "Supplier checks", "text": "Matching supplier invoices against purchase orders every week."}],
                         "tasks": [], "skip": []})
        rows = P.understand("anything", P.load_config())
        self.assertEqual([w["name"] for w in rows["work"]], ["Supplier checks"])

    def test_no_model_keeps_the_message_as_a_task(self):
        self.settings_file()
        self.fake_model(None)
        P.cmd_add(argparse.Namespace(text=["Stuck", "on", "fee", "tables", "today"], yes=True, no_scan=True))
        self.assertEqual(len(P.load_config()["tasks"]), 1)

    def test_new_rows_start_a_search_and_nothing_new_keeps_the_page(self):
        self.settings_file()
        scans, saved_scan = [], P.cmd_scan
        P.cmd_scan = lambda a: scans.append(a) or 0
        try:
            self.fake_model({"topics": [{"topic": "invoice matching", "importance": "High"}],
                             "work": [], "tasks": [], "skip": []})
            P.cmd_add(argparse.Namespace(text=["x"], yes=True, no_scan=False))
            self.assertEqual(len(scans), 1)                     # something new: search straight away
            P.cmd_add(argparse.Namespace(text=["x"], yes=True, no_scan=False))
            self.assertEqual(len(scans), 1)                     # already there: no search, page kept
        finally:
            P.cmd_scan = saved_scan

    def test_a_backup_is_made_before_writing(self):
        self.settings_file()
        self.fake_model({"topics": [], "work": [], "tasks": ["A task about contract dates"], "skip": []})
        P.cmd_add(argparse.Namespace(text=["x"], yes=True, no_scan=True))
        self.assertEqual(len(os.listdir(os.path.join(self.tmp, "settings_backups"))), 1)


class TestMixedTop(unittest.TestCase):
    def test_one_work_area_cannot_fill_the_shortlist(self):
        papers = [{"id": i, "Closest work": "AI agents"} for i in range(6)] + \
                 [{"id": 10, "Closest work": "Contract data recon"}, {"id": 11, "Closest work": "PE valuation"}]
        top = P.mixed_top(papers, 5)
        self.assertEqual([p["id"] for p in top], [0, 1, 10, 11, 2])   # 2 agents, then the other areas, then fill

    def test_without_meaning_the_order_is_kept(self):
        papers = [{"id": i, "Closest work": ""} for i in range(7)]
        self.assertEqual([p["id"] for p in P.mixed_top(papers, 5)], [0, 1, 2, 3, 4])


class TestAiGuards(Sandbox):
    def test_made_up_ids_and_bad_effort_are_dropped(self):
        saved = P.models.ask_json
        P.models.ask_json = lambda *a, **k: {"papers": [
            {"id": "2610.00001", "what": "real", "use_for_us": "x", "data_needed": "y", "effort": "XL"},
            {"id": "9999.99999", "what": "invented", "use_for_us": "x", "data_needed": "y", "effort": "S"}]}
        try:
            notes = P.explain([{"arXiv ID": "2610.00001", "Title": "t", "Abstract": "a"}])
        finally:
            P.models.ask_json = saved
        self.assertEqual(list(notes), ["2610.00001"])
        self.assertEqual(notes["2610.00001"]["effort"], "")

    def test_a_copied_work_label_is_not_kept_as_use_for_us(self):
        saved = P.models.ask_json
        P.models.ask_json = lambda *a, **k: {"papers": [
            {"id": "2610.00001", "what": "x", "use_for_us": "Contract data recon", "data_needed": "y", "effort": "M"}]}
        try:
            notes = P.explain([{"arXiv ID": "2610.00001", "Title": "t", "Abstract": "a", "Closest work": "Contract data recon"}],
                              {"Contract data recon": "Extracting fields from contracts"})
        finally:
            P.models.ask_json = saved
        self.assertEqual(notes["2610.00001"]["use_for_us"], "")


class TestPilots(Sandbox):
    def pick(self, pid, **kw):
        a = dict(id=pid, quarter="2026Q4", owner="Ekta", plan="", title="A paper", replace=False)
        a.update(kw)
        return P.cmd_pick(argparse.Namespace(**a))

    def test_one_pilot_per_quarter(self):
        self.pick("2610.00001")
        with self.assertRaises(SystemExit):
            self.pick("2610.00002")

    def test_adopted_needs_a_metric_and_then_counts_for_the_kra(self):
        self.pick("2610.00001")
        with self.assertRaises(SystemExit):
            P.cmd_result(argparse.Namespace(id="2610.00001", outcome="adopted", note="helped", metric=None))
        P.cmd_result(argparse.Namespace(id="2610.00001", outcome="adopted", note="helped", metric="7 -> 2"))
        out = os.path.join(self.tmp, "kra.xlsx")
        P.cmd_report(argparse.Namespace(out=out))
        overall = pd.read_excel(out, sheet_name="Overall")
        self.assertEqual(int(overall.set_index("KRA measure").loc["Adopted with documented business result", "Value"]), 1)


class TestQuality(Sandbox):
    def test_votes_give_the_share_of_cards_that_were_useful(self):
        month = P.datetime.now().strftime("%Y-%m")
        self.scan(f"Prospector_{month}")
        pd.DataFrame({"Month": [month, month, month], "Card": [1, 3, 3]}).to_excel(P.VOTES, index=False)
        q = P.quality()
        self.assertEqual(q["problems"], [])
        self.assertEqual(q["months"][0]["Cards with votes"], 2)
        self.assertEqual(q["months"][0]["Votes"], 3)

    def test_a_card_that_does_not_exist_is_reported(self):
        month = P.datetime.now().strftime("%Y-%m")
        self.scan(f"Prospector_{month}")
        pd.DataFrame({"Month": [month], "Card": [42]}).to_excel(P.VOTES, index=False)
        self.assertTrue(any("no card 42" in p for p in P.quality()["problems"]))


import ask_agent as AA  # noqa: E402


def paper(pid, title, abstract, days_ago=3):
    d = (pd.Timestamp.now() - pd.Timedelta(days=days_ago)).strftime("%Y-%m-%d")
    return {"arXiv ID": pid, "Title": title, "Abstract": abstract, "Published": d, "Categories": "cs.CL",
            "Link": f"https://arxiv.org/abs/{pid}"}


class AgentSandbox(Sandbox):
    """A sandbox with stored papers, a fake live arXiv search and a scripted AI (no tests of its own)."""

    def setUp(self):
        super().setUp()
        P.save_pool([paper("2611.00001", "Hallucination detection for financial reports",
                           "We detect hallucination in LLM financial report answers."),
                     paper("2611.00002", "Option pricing with neural networks", "Options volatility surface pricing."),
                     paper("2601.00003", "ESG report extraction", "Old paper on ESG report extraction.", 200)])
        self.saved_ask, self.saved_live = P.models.ask_json, AA.live_search
        self.calls, self.searched = [], []
        self.live = [paper("2605.00010", "LLM extraction of ESG report metrics", "ESG report metric extraction."),
                     paper("2605.00011", "ESG disclosure analysis with RAG", "ESG report retrieval."),
                     paper("2605.00012", "Benchmark for ESG report QA", "ESG report question answering.")]

        def live(words, days=365, n=40):
            self.searched.append(words)
            P.save_pool(self.live)
            return self.live
        AA.live_search = live

    def tearDown(self):
        P.models.ask_json, AA.live_search = self.saved_ask, self.saved_live
        super().tearDown()

    question_reply = None   # step 0 (input check): None = the AI gives no usable reply, so it is skipped
    claims_reply = None     # claim check: dict, or a function of the paper IDs; None = no checks

    def script(self, *replies):
        """The fake AI gives these replies in order (a function gets the candidate IDs)."""
        it = iter(replies)
        self.prompts = []

        def fake(role, prompt, system=None, temperature=0.0):
            self.prompts.append((role, prompt))
            if '"papers"' in prompt:   # explain() call: no notes
                return {"papers": []}
            if '"reading"' in prompt:   # input check (step 0)
                return self.question_reply
            ids = re.findall(r"^ID (\S+)", prompt, re.M)
            if '"checks"' in prompt:   # claim check against the abstracts
                c = self.claims_reply
                return (c(ids) if callable(c) else c) or {"checks": []}
            self.calls.append(ids)
            r = next(it)
            return r(ids) if callable(r) else r
        P.models.ask_json = fake



class TestAskAgent(AgentSandbox):
    """The agent loop with a scripted AI: who decides, what code allows, and what is shown."""

    def test_ai_decides_a_live_search_with_its_own_words_and_stops_at_the_goal(self):
        self.script({"useful": [], "next": "search_arxiv", "search_words": "ESG report extraction",
                     "thought": "Nothing on ESG here, so I search arXiv."},
                    lambda ids: {"useful": [{"id": i, "why": "about ESG reports"} for i in ids if i.startswith("2605")],
                                 "next": "search_arxiv", "search_words": "more esg", "thought": "good"})
        r = AA.ask("AI techniques for ESG reports")
        self.assertEqual(self.searched, ["esg report extraction"])            # the AI's words, cleaned
        self.assertEqual([s["tool"] for s in r["steps"]], ["search_recent", "search_arxiv"])
        self.assertEqual(r["steps"][0]["next_by"], "AI")
        self.assertEqual(r["steps"][1]["next_by"], "rule")                    # goal reached: code stops
        self.assertEqual({p["id"] for p in r["papers"]}, {"2605.00010", "2605.00011", "2605.00012"})
        self.assertTrue(os.path.exists(P.ASK_LOG))

    def test_made_up_ids_and_unknown_tools_are_not_accepted(self):
        self.script({"useful": [{"id": "9999.99999", "why": "invented"}], "next": "delete_everything",
                     "thought": "x"}, {"useful": [], "next": "stop", "thought": "x"},
                    {"useful": [], "next": "stop", "thought": "x"})
        r = AA.ask("hallucination detection in reports")
        self.assertNotIn("9999.99999", [p["id"] for p in r["papers"]])
        self.assertEqual(r["steps"][0]["next"], "search_stored")              # unknown tool -> next in order
        self.assertEqual(r["steps"][0]["next_by"], "rule")

    def test_the_agent_may_not_give_up_before_one_live_search(self):
        self.script({"useful": [], "next": "stop", "thought": "nothing"},
                    {"useful": [], "next": "stop", "thought": "nothing"})
        self.live = []
        r = AA.ask("gift card fraud prevention")
        self.assertEqual([s["tool"] for s in r["steps"]], ["search_recent", "search_arxiv"])
        self.assertIn("before giving up", r["steps"][0]["note"])
        self.assertEqual(r["papers"], [])
        self.assertIn("Nothing is made up", r["outcome"])

    def test_without_the_ai_it_goes_by_meaning_in_a_fixed_order(self):
        P.models.ask_json = lambda *a, **k: None
        r = AA.ask("ESG report extraction")
        self.assertEqual(r["ai"], "off")
        self.assertTrue(all(s["by"] == "rule" for s in r["steps"]))
        self.assertEqual([s["tool"] for s in r["steps"]][:2], ["search_recent", "search_stored"])

    def test_a_step_is_never_repeated_with_the_same_words(self):
        self.script(*[{"useful": [], "next": "search_arxiv", "search_words": "same words", "thought": "again"}] * 4)
        self.live = []
        r = AA.ask("something rare")
        searched = [(s["tool"], s["words"]) for s in r["steps"]]
        self.assertEqual(len(searched), len(set(searched)))
        self.assertLessEqual(len(r["steps"]), AA.ROUNDS)

    def test_web_app_answers_and_leaves_the_settings_alone(self):
        P.write_settings(P.load_config(), P.SETTINGS)
        before = os.path.getmtime(P.SETTINGS)
        sys.path.insert(0, SRC)
        import app as webapp
        from fastapi.testclient import TestClient
        self.script(lambda ids: {"useful": [{"id": i, "why": "fits"} for i in ids[:1]], "next": "stop", "thought": "ok"})
        r = TestClient(webapp.app).post("/api/ask", json={"text": "hallucination detection in financial reports"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["steps"][0]["tool"], "search_recent")
        self.assertEqual(os.path.getmtime(P.SETTINGS), before)
        self.assertEqual(TestClient(webapp.app).post("/api/ask", json={"text": "x"}).status_code, 422)

    def test_when_arxiv_is_down_the_agent_searches_stored_papers_and_does_not_retry_arxiv(self):
        def down(words, days=365, n=40):
            self.searched.append(words)
            raise OSError("HTTP Error 429")
        AA.live_search = down
        self.script(*[{"useful": [], "next": "search_arxiv", "search_words": f"words {i}", "thought": "x"}
                      for i in range(4)])
        r = AA.ask("something rare")
        self.assertEqual([s["tool"] for s in r["steps"]], ["search_recent", "search_arxiv", "search_stored"])
        self.assertEqual(len(self.searched), 1)                                  # arXiv tried once only
        self.assertIn("arXiv is not reachable", r["steps"][1]["note"])
        self.assertEqual(r["steps"][1]["next_by"], "rule")

    # live steps --------------------------------------------------------------------------------------------
    def test_steps_arrive_one_by_one_before_the_answer(self):
        self.script({"useful": [], "next": "search_arxiv", "search_words": "ESG report extraction", "thought": "x"},
                    lambda ids: {"useful": [{"id": i, "why": "about ESG"} for i in ids if i.startswith("2605")],
                                 "next": "stop", "thought": "good"})
        events = list(AA.ask_steps("AI techniques for ESG reports"))
        kinds = [e["type"] for e in events]
        self.assertEqual(kinds[-1], "done")
        self.assertEqual(kinds.count("check"), 1)
        self.assertLess(kinds.index("check"), kinds.index("step"))              # step 0 comes first
        steps = [e["step"] for e in events if e["type"] == "step"]
        self.assertEqual(steps, events[-1]["result"]["steps"])                   # same steps as the final answer
        self.assertTrue(all("next" in s and "next_by" in s for s in steps))      # each step is complete when sent
        self.assertIn("status", kinds)

    def test_stream_endpoint_sends_one_event_per_line_and_frees_the_agent(self):
        sys.path.insert(0, SRC)
        import app as webapp
        from fastapi.testclient import TestClient
        c = TestClient(webapp.app)
        for _ in range(2):   # a second question works: the agent was freed after the first
            self.script(lambda ids: {"useful": [{"id": i, "why": "fits"} for i in ids[:1]], "next": "stop",
                                     "thought": "ok"})
            r = c.post("/api/ask/stream", json={"text": "hallucination detection in financial reports", "user": "Ekta"})
            self.assertEqual(r.status_code, 200)
            events = [json.loads(line) for line in r.text.splitlines() if line.strip()]
            self.assertEqual(events[-1]["type"], "done")
            self.assertEqual(events[-1]["result"]["user"], "Ekta")
        webapp._busy["since"] = __import__("time").time()   # another question is running
        self.assertEqual(c.post("/api/ask/stream", json={"text": "anything at all"}).status_code, 409)
        webapp._busy["since"] = None
        self.assertEqual(c.post("/api/ask/stream", json={"text": "x"}).status_code, 422)

    def test_a_browser_that_leaves_mid_run_does_not_keep_the_agent_busy(self):
        import time as _t
        import summary as S
        sys.path.insert(0, SRC)
        import app as webapp
        from fastapi.testclient import TestClient
        self.script(lambda ids: {"useful": [{"id": i, "why": "fits"} for i in ids[:1]], "next": "stop", "thought": "ok"})
        with TestClient(webapp.app).stream("POST", "/api/ask/stream",
                                           json={"text": "hallucination detection in reports", "user": "Ekta"}) as r:
            next(r.iter_lines())   # read the first event, then the browser is gone
        for _ in range(50):
            if webapp._busy["since"] is None:
                break
            _t.sleep(0.1)
        self.assertIsNone(webapp._busy["since"])                                 # free for the next question
        self.assertEqual(S.weekly("Ekta")["totals"]["questions"], 1)             # and the run was still saved

    # input check -------------------------------------------------------------------------------------------
    def test_input_check_spots_a_typo_and_searches_the_corrected_question(self):
        self.question_reply = {"reading": "Ways to detect hallucinations in financial reports", "issue": "typo",
                               "note": "'halucinaton' looks like 'hallucination'.",
                               "search_text": "hallucination detection in financial reports"}
        self.script(lambda ids: {"useful": [{"id": i, "why": "fits"} for i in ids[:1]], "next": "stop", "thought": "ok"})
        r = AA.ask("halucinaton detecton in finacial reports")
        self.assertEqual(r["check"]["issue"], "typo")
        self.assertIn("hallucination", r["check"]["note"])
        self.assertEqual(r["question"], "halucinaton detecton in finacial reports")     # what the user typed is kept
        judge_prompts = [p for role, p in self.prompts if "CANDIDATE PAPERS" in p]
        self.assertIn("QUESTION: hallucination detection in financial reports", judge_prompts[0])
        self.assertIn("2611.00001", [p["id"] for p in r["papers"]])

    def test_input_check_never_changes_the_question_when_nothing_is_wrong_or_the_ai_is_off(self):
        self.question_reply = {"reading": "ESG", "issue": "none", "note": "ignored", "search_text": "something else"}
        self.script(lambda ids: {"useful": [], "next": "stop", "thought": "x"}, {"useful": [], "next": "stop"})
        r = AA.ask("ESG report extraction")
        self.assertEqual((r["check"]["search_text"], r["check"]["note"]), ("ESG report extraction", ""))
        P.models.ask_json = lambda *a, **k: None
        r = AA.ask("ESG report extraction", use_ai=False)
        self.assertEqual(r["check"]["by"], "rule")
        self.assertIn("searched as typed", r["check"]["skipped"])

    def test_code_finds_typos_the_ai_missed_even_with_the_ai_off(self):
        P.models.ask_json = lambda *a, **k: None
        r = AA.ask("halucination detection in financial reports", use_ai=False)
        self.assertEqual(r["check"]["issue"], "typo")
        self.assertEqual(r["check"]["search_text"], "hallucination detection in financial reports")
        self.assertEqual(r["check"]["spelling"], [{"typed": "halucination", "suggested": "hallucination"}])
        r = AA.ask("ESG report extraction", use_ai=False)                       # correct words: nothing flagged
        self.assertEqual(r["check"]["issue"], "none")

    # claim check and confidence ----------------------------------------------------------------------------
    def test_a_claim_the_abstract_does_not_support_gets_low_confidence(self):
        self.script({"useful": [], "next": "search_arxiv", "search_words": "esg report", "thought": "x"},
                    lambda ids: {"useful": [{"id": i, "why": "about ESG"} for i in ids if i.startswith("2605")],
                                 "next": "stop", "thought": "good"})
        self.claims_reply = lambda ids: {"checks": [
            {"id": "2605.00011", "supported": "no", "problem": "The abstract is about retrieval, not extraction."},
            {"id": "2605.00010", "supported": "yes", "problem": ""},
            {"id": "2605.00012", "supported": "yes", "problem": ""},
            {"id": "9999.99999", "supported": "yes"}]}                             # made up: ignored
        r = AA.ask("AI techniques for ESG reports")
        by = {p["id"]: p for p in r["papers"]}
        bad = by["2605.00011"]
        self.assertTrue(bad["low_confidence"])
        self.assertEqual(bad["confidence_label"], "Low")
        self.assertLess(bad["confidence"], AA.verify.LOW_CONFIDENCE)
        self.assertIn("retrieval", bad["claim_check"]["problem"])
        self.assertFalse(by["2605.00010"]["low_confidence"])
        self.assertEqual(r["papers"][-1]["id"], "2605.00011")                     # sorted by confidence
        self.assertIn("low confidence", r["outcome"])
        self.assertNotIn("9999.99999", by)

    def test_a_paper_on_a_neighbouring_topic_is_low_even_when_its_summary_is_right(self):
        self.question_reply = {"reading": "Use AI to improve our own code so servers cost less", "issue": "none",
                               "note": "", "search_text": "ESG report extraction"}
        self.script({"useful": [], "next": "search_arxiv", "search_words": "esg report", "thought": "x"},
                    lambda ids: {"useful": [{"id": i, "why": "fits"} for i in ids if i.startswith("2605")],
                                 "next": "search_arxiv", "search_words": "more", "thought": "more"})
        self.claims_reply = lambda ids: {"checks": [
            {"id": "2605.00010", "supported": "yes", "answers": "yes", "answers_why": "on point"},
            {"id": "2605.00011", "supported": "yes", "answers": "no",
             "answers_why": "It is about the cost of running AI models, not improving code."},
            {"id": "2605.00012", "supported": "yes", "answers": "partly", "answers_why": "related"}]}
        r = AA.ask("AI to make our code cheaper to run")
        by = {p["id"]: p for p in r["papers"]}
        self.assertFalse(by["2605.00010"]["low_confidence"])
        self.assertGreater(by["2605.00010"]["confidence"], by["2605.00012"]["confidence"])
        self.assertEqual(by["2605.00011"]["confidence_label"], "Low")
        self.assertTrue(by["2605.00011"]["low_confidence"])
        self.assertEqual(by["2605.00012"]["confidence_label"], "Medium")
        self.assertEqual([p["id"] for p in r["papers"]][0], "2605.00010")
        prompts = [p for _, p in self.prompts]
        self.assertTrue(any("WHAT THE USER WANTS: Use AI to improve our own code" in p and '"answers"' in p
                            for p in prompts))                                    # claim check sees step 0's reading
        self.assertTrue(any("WHAT THE USER WANTS: Use AI to improve our own code" in p and "CANDIDATE PAPERS" in p
                            for p in prompts))                                    # and so does the search step
        self.assertEqual(r["steps"][-1]["ai_wanted"], "search_arxiv 'more'")     # the rule stopped it: shown

    def test_only_the_most_confident_papers_are_shown_and_none_answering_is_said(self):
        self.script({"useful": [], "next": "search_arxiv", "search_words": "esg report", "thought": "x"},
                    lambda ids: {"useful": [{"id": i, "why": "fits"} for i in ids if i.startswith("2605")],
                                 "next": "stop", "thought": "ok"})
        self.claims_reply = lambda ids: {"checks": [{"id": i, "supported": "yes", "answers": "no"} for i in ids]}
        r = AA.ask("AI techniques for ESG reports")
        self.assertLessEqual(len(r["papers"]), AA.SHOW)
        self.assertIn("None of them really answers your question", r["outcome"])

    def test_confidence_rule(self):
        V = AA.verify
        yes = {"supported": "yes", "answers": "yes"}
        self.assertEqual(V.confidence(0.80, yes)[:2], (100, "High"))
        self.assertEqual(V.confidence(0.50, {"supported": "no", "answers": "no"})[:2], (0, "Low"))
        self.assertEqual(V.confidence(0.65, {"supported": "partly", "answers": "yes"})[1], "Medium")
        # a faithful summary of a paper that does not answer the question is never trusted
        self.assertEqual(V.confidence(0.80, {"supported": "yes", "answers": "no"})[1], "Low")
        self.assertEqual(V.confidence(0.80, {"supported": "yes", "answers": "partly"})[1], "Medium")
        self.assertEqual(V.confidence(0.80, {"supported": "yes"})[1], "Medium")      # not said: treated as partly
        self.assertLessEqual(V.confidence(0.90, None, ai_checked=False)[0], V.LOW_CONFIDENCE + 10)  # never High
        self.assertEqual(V.confidence(0.80, yes, question_issue="vague")[1], "Medium")  # vague: capped
        P.models.ask_json = lambda *a, **k: {"checks": [{"id": "1", "supported": "yes", "problem": ""}]}
        c = V.check_claims("q", [{"id": "1", "abstract": "We cut errors by 12%.", "what": "cuts errors by 45%"}])
        self.assertEqual(c["1"]["supported"], "partly")                          # a number not in the abstract
        self.assertIn("45", c["1"]["problem"])


class TestArxivRules(Sandbox):
    """The Ask agent keeps to arXiv's API rules: 1 request every 3 s, one retry after 429, repeats from memory."""

    def setUp(self):
        super().setUp()
        import urllib.error
        self.err = urllib.error.HTTPError
        self.saved_net = (AA.urllib.request.urlopen, AA.ARXIV_GAP, AA.ARXIV_RETRY_WAIT, AA.time.sleep)
        self.calls, self.slept, self.fail = [], [], 0
        AA._arxiv_cache.clear()
        AA._arxiv_last[0] = 0.0
        feed = open(FEED, "rb").read()
        test = self

        class Resp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return feed

        def urlopen(req, timeout=30, context=None):
            test.calls.append(req.full_url)
            if test.fail:
                test.fail -= 1
                raise test.err(req.full_url, 429, "Too Many Requests", {"Retry-After": "2"}, None)
            return Resp()
        AA.urllib.request.urlopen = urlopen
        AA.time.sleep = lambda s: self.slept.append(round(s, 1))

    def tearDown(self):
        AA.urllib.request.urlopen, AA.ARXIV_GAP, AA.ARXIV_RETRY_WAIT, AA.time.sleep = self.saved_net
        AA._arxiv_cache.clear()
        super().tearDown()

    def test_requests_are_spaced_three_seconds_apart(self):
        AA.arxiv("https://export.arxiv.org/api/query?a=1")
        AA.arxiv("https://export.arxiv.org/api/query?a=2")
        self.assertEqual(len(self.calls), 2)
        self.assertTrue(self.slept and 2.5 <= self.slept[-1] <= 3.0)              # waited before the second

    def test_a_429_is_retried_once_after_the_wait_arxiv_asks_for(self):
        self.fail = 1
        papers = AA.arxiv("https://export.arxiv.org/api/query?b=1")
        self.assertTrue(papers)
        self.assertEqual(len(self.calls), 2)
        self.assertIn(2.0, self.slept)                                            # Retry-After: 2
        self.fail = 2
        with self.assertRaises(self.err):                                         # still refused: the agent falls back
            AA.arxiv("https://export.arxiv.org/api/query?b=2")

    def test_the_same_search_again_is_answered_from_memory(self):
        first = AA.arxiv("https://export.arxiv.org/api/query?c=1")
        again = AA.arxiv("https://export.arxiv.org/api/query?c=1")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual([p["arXiv ID"] for p in first], [p["arXiv ID"] for p in again])


class TestWeeklySummary(AgentSandbox):
    """Every question is saved for its user in users.json; the summary is that user's week."""

    def ask_as(self, user, question, answered=True):
        if answered:
            self.script(lambda ids: {"useful": [{"id": i, "why": "fits"} for i in ids[:1]], "next": "stop",
                                     "thought": "ok"})
        else:
            self.live = []
            self.script({"useful": [], "next": "stop", "thought": "x"}, {"useful": [], "next": "stop", "thought": "x"},
                        {"useful": [], "next": "stop", "thought": "x"}, {"useful": [], "next": "stop", "thought": "x"})
        return AA.ask(question, user=user)

    def test_questions_are_saved_per_user_and_summed_for_the_week(self):
        import summary as S
        self.ask_as("Ekta", "hallucination detection in financial reports")
        self.ask_as("Ekta", "gift card fraud prevention", answered=False)
        self.ask_as("Ravi", "option pricing with neural networks")
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "users.json")))
        month = pd.Timestamp.now().strftime("%Y-%m")
        w = S.weekly("ekta", votes={month: {"EKTA": ["2611.00001"], "Ravi": ["2611.00002"]}})   # name: any case
        self.assertEqual((w["totals"]["questions"], w["totals"]["answered"], w["totals"]["unanswered"]), (2, 1, 1))
        self.assertEqual(w["unanswered"][0]["question"], "gift card fraud prevention")
        self.assertEqual(w["totals"]["votes"], 1)
        self.assertEqual(w["team"], {"people": 2, "questions": 3})
        self.assertNotIn("option pricing", json.dumps(w["questions"]))            # only this user's questions
        self.assertTrue(any("honestly unanswered" in h for h in w["highlights"]))
        self.assertEqual(S.users(), {"ekta": "Ekta", "ravi": "Ravi"})

    def test_other_weeks_are_kept_apart_and_the_web_app_serves_the_summary(self):
        import summary as S
        from datetime import datetime, timedelta
        r = self.ask_as("Ekta", "hallucination detection in financial reports")
        S.record(r, when=datetime.now() - timedelta(days=14))                    # the same question two weeks ago
        w = S.weekly("Ekta")
        self.assertEqual(w["totals"]["questions"], 1)
        self.assertEqual(len(w["weeks"]), 2)
        old = S.weekly("Ekta", week=w["weeks"][1])
        self.assertEqual(old["totals"]["questions"], 1)
        sys.path.insert(0, SRC)
        import app as webapp
        from fastapi.testclient import TestClient
        saved, webapp.HOME = webapp.HOME, self.tmp
        try:
            c = TestClient(webapp.app)
            got = c.get("/api/summary", params={"user": "Ekta"})
            self.assertEqual(got.status_code, 200)
            self.assertEqual(got.json()["totals"]["questions"], 1)
            self.assertEqual(c.get("/api/summary", params={"user": " "}).status_code, 400)
            self.assertEqual(c.get("/api/summary", params={"user": "Ekta", "week": "last"}).status_code, 400)
            self.assertEqual(c.get("/api/summary/users").json(), {"users": ["Ekta"]})
        finally:
            webapp.HOME = saved

    def test_no_name_counts_for_the_team_only(self):
        import summary as S
        self.ask_as("", "hallucination detection in financial reports")
        self.assertEqual(S.users(), {})
        self.assertEqual(S.weekly("Someone")["team"]["questions"], 1)

    def test_a_broken_users_file_never_stops_the_agent(self):
        import summary as S
        with open(S.users_file(), "w") as f:
            f.write("{not json")
        r = self.ask_as("Ekta", "hallucination detection in financial reports")
        self.assertTrue(r["papers"])
        self.assertEqual(S.weekly("Ekta")["totals"]["questions"], 1)


class TestRadarPage(Sandbox):
    def build(self, **cfg_extra):
        out = self.scan()
        if cfg_extra:
            saved = P.load_config
            P.load_config = lambda: {**saved(), **cfg_extra}
        try:
            path, data = B.build(out + ".xlsx", out=os.path.join(self.tmp, "radar.html"))
        finally:
            if cfg_extra:
                P.load_config = saved
        return path, data

    def test_page_has_data_and_no_placeholder(self):
        path, data = self.build()
        html = read(path)
        self.assertNotIn("__DATA__", html)
        self.assertEqual(len(data["short"]), 5)
        self.assertIn("run", data)

    def render(self, path, in_claude=False):
        probe = os.path.join(self.tmp, "probe.html")
        stub = "window.claude={use:()=>Promise.resolve(null)};" if in_claude else ""  # claude.ai viewer, signed out
        with open(probe, "w") as f:
            f.write('<!doctype html><html><head><meta charset=utf-8><script>' + stub + 'window.onerror=(m,s,l)=>'
                    '{document.documentElement.setAttribute("data-err",m+" @"+l)}</script></head><body>'
                    + read(path) +
                    '<script>setTimeout(()=>{const b=document.body;'
                    'b.setAttribute("data-cards",document.querySelectorAll(".card").length);'
                    'b.setAttribute("data-votes",document.querySelectorAll("a.vote").length);'
                    'b.setAttribute("data-run",document.getElementById("run-line").textContent.length>0)},800)'
                    '</script></body></html>')
        dom = subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--virtual-time-budget=4000",
                              "--dump-dom", "file://" + probe], capture_output=True, text=True, timeout=60).stdout
        return dict(re.findall(r'data-(err|cards|votes|run)="([^"]*)"', dom))

    @unittest.skipUnless(os.path.exists(CHROME), "Google Chrome not installed")
    def test_page_renders_every_card_without_errors(self):
        """The missing-cards bug: the page must draw every shortlist card."""
        path, data = self.build()
        got = self.render(path)
        self.assertNotIn("err", got, got.get("err"))
        self.assertEqual(int(got["cards"]), len(data["short"]))
        self.assertEqual(got["run"], "true")
        self.assertEqual(int(got["votes"]), 0)   # no form link: no vote buttons outside claude.ai

    @unittest.skipUnless(os.path.exists(CHROME), "Google Chrome not installed")
    def test_page_renders_every_card_inside_claude_ai_too(self):
        """Opened on claude.ai the page takes the built-in voting path; it must still draw every card."""
        path, data = self.build()
        got = self.render(path, in_claude=True)
        self.assertNotIn("err", got, got.get("err"))
        self.assertEqual(int(got["cards"]), len(data["short"]))

    @unittest.skipUnless(os.path.exists(CHROME), "Google Chrome not installed")
    def test_vote_buttons_open_the_form_when_one_is_set(self):
        path, data = self.build(vote_form="https://forms.office.com/r/example")
        got = self.render(path)
        self.assertGreaterEqual(int(got["votes"]), len(data["short"]))


if __name__ == "__main__":
    unittest.main()
