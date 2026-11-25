"""Stream authenticated joint artifacts into the semantic validator's object tree."""
import hashlib
import os
from pathlib import Path
import stat


def require_job_scope() -> None:
    declared = Path(os.environ.get("CX_EXPERIMENT_MEMORY_SCOPE", ""))
    current = next((Path(line[3:]) for line in Path("/proc/self/cgroup").read_text().splitlines()
                    if line.startswith("0::")), None)
    if (current is None or not declared.is_absolute() or declared == Path("/") or ".." in declared.parts
            or not current.is_relative_to(declared)):
        raise ValueError("joint graph conversion requires its whole-job cgroup")
    root = Path("/sys/fs/cgroup") / declared.relative_to("/")
    maximum = (root / "memory.max").read_text().strip()
    if (not maximum.isdecimal() or int(maximum) <= 0 or (root / "memory.oom.group").read_text().strip() != "1"
            or (root / "memory.swap.max").read_text().strip() != "0"):
        raise ValueError("joint graph conversion requires finite memory.max, group OOM and disabled swap")


def read_job_document(path, *, max_depth=None):
    """Keep one decoded graph, hashing the exact chunks consumed by the parser."""
    import ijson
    from ijson.common import ObjectBuilder

    require_job_scope()
    path = Path(path).absolute()
    if ".." in path.parts or path.resolve(strict=True) != path:
        raise ValueError("joint input must use an original path without links")
    digest = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("joint input must be a regular file")

        class DigestReader:
            count = 0

            def read(self, size):
                block = stream.read(size)
                self.count += len(block)
                if self.count > before.st_size:
                    raise ValueError("joint input grew while reading")
                digest.update(block)
                return block

        class UniqueFields(dict):
            def __setitem__(self, key, value):
                if key in self:
                    raise ValueError("duplicate JSON key")
                super().__setitem__(key, value)

        builder, reader, depth = ObjectBuilder(map_type=UniqueFields), DigestReader(), 0
        try:
            for event, value in ijson.basic_parse(reader):
                depth += int(event in ("start_map", "start_array")) - int(event in ("end_map", "end_array"))
                if max_depth is not None and depth > max_depth:
                    raise ValueError("JSON nesting exceeds limit")
                builder.event(event, value)
        except ijson.JSONError as error:
            raise ValueError("invalid joint JSON original") from error
        def identity(st):
            return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns
        if (reader.count != before.st_size or identity(before) != identity(os.fstat(fd))
                or identity(before) != identity(path.stat(follow_symlinks=False))):
            raise ValueError("joint input changed during parsing")
    return builder.value, digest.hexdigest()
