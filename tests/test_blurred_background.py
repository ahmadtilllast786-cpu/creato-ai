import os
import pytest
import cv2

main = pytest.importorskip("main")


class TestBlurredBackgroundFilter:
    def test_filter_exact_structure(self):
        filt = main.build_blurred_background_filter(out_w=1080, out_h=1920, dim=False)
        assert "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:5[bg]" in filt
        assert "[0:v]scale=1080:-2[fg]" in filt
        assert "[bg][fg]overlay=(W-w)/2:(H-h)/2" in filt

    def test_filter_with_dimming(self):
        filt = main.build_blurred_background_filter(out_w=1080, out_h=1920, dim=True)
        assert "boxblur=20:5,eq=brightness=-0.05[bg]" in filt
        assert "[0:v]scale=1080:-2[fg]" in filt
        assert "[bg][fg]overlay=(W-w)/2:(H-h)/2" in filt

    def test_filter_dimensions_are_strictly_even(self):
        for w, h in [(1080, 1920), (1079, 1919), (720, 1280), (1080, 1080)]:
            filt = main.build_blurred_background_filter(out_w=w, out_h=h)
            bg_scale = filt.split("[0:v]scale=")[1].split(":force_original_aspect_ratio")[0]
            sw, sh = [int(v) for v in bg_scale.split(":")]
            assert sw % 2 == 0
            assert sh % 2 == 0
            crop_part = filt.split(",crop=")[1].split(",boxblur=")[0]
            cw, ch = [int(v) for v in crop_part.split(":")]
            assert cw % 2 == 0
            assert ch % 2 == 0

    def test_landscape_and_tall_sources(self):
        filt_land = main.build_blurred_background_filter(1080, 1920, orig_w=1920, orig_h=1080)
        assert "[0:v]scale=1080:-2[fg]" in filt_land

        filt_tall = main.build_blurred_background_filter(1080, 1920, orig_w=1080, orig_h=2400)
        assert "[0:v]scale=-2:1920[fg]" in filt_tall


class TestProcessVideoToVertical:
    def test_end_to_end_blurred_reframing(self, tmp_path):
        source = "demo-openshorts.mp4"
        if not os.path.exists(source):
            pytest.skip("demo-openshorts.mp4 not found")

        clip_src = str(tmp_path / "test_src.mp4")
        out_vert = str(tmp_path / "test_out_vertical.mp4")

        cut_cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", "0", "-t", "1", "-i", source,
            "-c:v", "libx264", "-preset", "ultrafast", "-an",
            clip_src
        ]
        main.run_ffmpeg_command(cut_cmd)
        assert os.path.exists(clip_src)

        res = main.process_video_to_vertical(clip_src, out_vert)
        assert res is True
        assert os.path.exists(out_vert)
        assert os.path.getsize(out_vert) > 0

        cap = cv2.VideoCapture(out_vert)
        ret, frame = cap.read()
        cap.release()
        assert ret is True
        h, w, c = frame.shape
        assert w == 1080
        assert h == 1920
        assert c == 3

        top_bar = frame[100, 540]
        assert top_bar.mean() > 5
