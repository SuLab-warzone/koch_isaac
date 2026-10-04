"""Optional rollout artifacts. Disabled means no directory or output-file creation."""
from datetime import datetime
import json
from pathlib import Path


def should_save_output(requested, num_envs):
    return bool(requested and num_envs == 1)


class RunOutput:
    def __init__(self, args, root, label):
        self.enabled = should_save_output(args.save_output, args.num_envs)
        self.path = None
        self.log = self.video = None
        self.frames = 0
        self.snapshot = args.snapshot if self.enabled else None
        if self.enabled:
            self.path = (args.output_dir or Path(root)/"outputs"/(label+"_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))).expanduser().resolve()
            self.path.mkdir(parents=True, exist_ok=True)
            self.log = (self.path/"rollout.jsonl").open("w")
            if args.video:
                import imageio.v2 as imageio
                self.video = imageio.get_writer(str(self.path/"front.mp4"), fps=30, codec="libx264")
            print(f"Rollout output: {self.path}", flush=True)

    def record(self, event, **values):
        if self.log is not None:
            self.log.write(json.dumps({"event":event, **values})+"\n")
            self.log.flush()

    def frame(self, rgb):
        if not self.enabled: return
        from PIL import Image
        if self.frames == 0: Image.fromarray(rgb).save(self.path/"first_frame.png")
        if self.video is not None: self.video.append_data(rgb)
        self.frames += 1

    def finish(self, summary, last_rgb=None):
        self.record("summary", **summary)
        if self.video is not None: self.video.close(); self.video=None
        if self.log is not None: self.log.close(); self.log=None
        if self.enabled:
            (self.path/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
            if last_rgb is not None:
                from PIL import Image
                Image.fromarray(last_rgb).save(self.path/"last_frame.png")
                if self.snapshot:
                    self.snapshot.parent.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(last_rgb).save(self.snapshot)
        print("Evaluation summary: "+json.dumps(summary),flush=True)
