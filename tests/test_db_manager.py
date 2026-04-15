import os
import sys
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_manager import DatabaseManager

def test_database_manager():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = DatabaseManager(db_path)
        
        print("测试 1: 添加受试者")
        result = db.add_subject("S001", "张三", age=25, gender="男", psqi_score=10.5)
        assert result == True, "添加受试者失败"
        print("✓ 添加受试者成功")
        
        print("\n测试 2: 查询受试者")
        subject = db.get_subject("S001")
        assert subject is not None, "查询受试者失败"
        assert subject['name'] == "张三", "受试者姓名不匹配"
        print(f"✓ 查询受试者成功: {subject}")
        
        print("\n测试 3: 更新受试者")
        result = db.update_subject("S001", age=26, psqi_score=12.0)
        assert result == True, "更新受试者失败"
        subject = db.get_subject("S001")
        assert subject['age'] == 26, "年龄更新失败"
        print(f"✓ 更新受试者成功: {subject}")
        
        print("\n测试 4: 查询所有受试者")
        db.add_subject("S002", "李四", age=30, gender="女")
        subjects = db.get_all_subjects()
        assert len(subjects) == 2, "受试者数量不正确"
        print(f"✓ 查询所有受试者成功: 共 {len(subjects)} 人")
        
        print("\n测试 5: 搜索受试者")
        results = db.search_subjects("张")
        assert len(results) == 1, "搜索结果不正确"
        print(f"✓ 搜索受试者成功: 找到 {len(results)} 人")
        
        print("\n测试 6: 添加 EEG 会话")
        session_id = db.add_eeg_session("S001", "/path/to/eeg.edf", 256.0, 19, 3600.0)
        assert session_id > 0, "添加 EEG 会话失败"
        print(f"✓ 添加 EEG 会话成功: session_id = {session_id}")
        
        print("\n测试 7: 查询受试者的 EEG 会话")
        sessions = db.get_subject_sessions("S001")
        assert len(sessions) == 1, "EEG 会话数量不正确"
        print(f"✓ 查询 EEG 会话成功: 共 {len(sessions)} 个会话")
        
        print("\n测试 8: 添加 SWS 结果")
        sws_id = db.add_sws_result(session_id, 1200.0, 150.5, "[]")
        assert sws_id > 0, "添加 SWS 结果失败"
        print(f"✓ 添加 SWS 结果成功: sws_id = {sws_id}")

        print("\n测试 8.1: 更新同一会话的 SWS 结果")
        updated_sws_id = db.upsert_sws_result(session_id, 1500.0, 180.5, '[{\"epoch_index\": 0}]')
        assert updated_sws_id == sws_id, "SWS 结果应更新同一条记录"
        print(f"✓ 更新 SWS 结果成功: sws_id = {updated_sws_id}")
        
        print("\n测试 9: 添加音乐结果")
        music_id = db.add_music_result(session_id, "/path/to/music.mid", "/path/to/music.wav", 440.0, 120.0, 180.0)
        assert music_id > 0, "添加音乐结果失败"
        print(f"✓ 添加音乐结果成功: music_id = {music_id}")

        print("\n测试 9.1: 更新同一会话的音乐结果")
        updated_music_id = db.upsert_music_result(session_id, "/path/to/music_v2.mid", "/path/to/music_v2.wav", 442.0, 90.0, 300.0)
        assert updated_music_id == music_id, "音乐结果应更新同一条记录"
        print(f"✓ 更新音乐结果成功: music_id = {updated_music_id}")
        
        print("\n测试 10: 查询最新结果")
        latest = db.get_latest_results("S001")
        assert latest is not None, "查询最新结果失败"
        assert latest['sws_duration'] == 1500.0, "SWS 时长不匹配"
        assert latest['avg_tempo'] == 90.0, "音乐节奏不匹配"
        print(f"✓ 查询最新结果成功: {latest}")
        
        print("\n测试 11: 删除受试者")
        result = db.delete_subject("S001")
        assert result == True, "删除受试者失败"
        subject = db.get_subject("S001")
        assert subject is None, "受试者未删除"
        print("✓ 删除受试者成功")
        
        print("\n测试 12: 重复 ID 约束")
        db.add_subject("S003", "王五")
        result = db.add_subject("S003", "赵六")
        assert result == False, "重复 ID 应该失败"
        print("✓ 重复 ID 约束验证成功")
        
        print("\n" + "="*50)
        print("所有测试通过！重构成功！")
        print("="*50)

if __name__ == "__main__":
    test_database_manager()
