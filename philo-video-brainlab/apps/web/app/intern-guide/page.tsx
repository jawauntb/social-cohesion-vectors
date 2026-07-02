import Link from "next/link";

const coreColumns = [
  "video_id",
  "platform",
  "url",
  "video_file_url",
  "title",
  "creator",
  "views",
  "likes",
  "comments",
  "shares",
  "saves",
  "reposts",
  "avg_retention",
  "completion_rate",
  "evidence_url",
  "screenshot_url",
  "notes",
];

export default function InternGuidePage() {
  return (
    <main>
      <h1>Intern data collection guide</h1>
      <p className="lede">
        Your job is to make a clean spreadsheet of old videos and visible engagement numbers. You
        do not need to understand the brain model. Accuracy and evidence matter more than speed.
      </p>

      <section className="guide-band">
        <h2>What You Are Collecting</h2>
        <p>
          Make one CSV per account or source, like <code>our_videos.csv</code>,{" "}
          <code>lectures_on_tap.csv</code>, or <code>met_museum.csv</code>. Each row is one video
          or post. We will compare our videos against 2-3 competitors or controls.
        </p>
        <p className="muted">
          Minimum useful row: public post link, actual video file link when you can get it,
          platform, title, creator/account, visible engagement numbers, and a screenshot or
          evidence link.
        </p>
      </section>

      <h2>Step-by-step</h2>
      <ol className="steps">
        <li>
          <strong>Pick one account.</strong>
          <span>Work source by source. Finish one account before mixing in another.</span>
        </li>
        <li>
          <strong>Choose a spread of posts.</strong>
          <span>Collect some low, medium, and high performers if you can tell from the numbers.</span>
        </li>
        <li>
          <strong>Open each post.</strong>
          <span>Copy the public post URL, title/caption if visible, platform, and creator/account name.</span>
        </li>
        <li>
          <strong>Save the actual video file.</strong>
          <span>
            The model cannot reliably watch an Instagram, TikTok, or YouTube page link. It needs
            the actual video file. First look for the original export in our shared folders. If the
            creator/account gives a download option, use that. Upload the file to the same Google
            Drive folder as the CSV and paste the shared file link into <code>video_file_url</code>.
          </span>
        </li>
        <li>
          <strong>Record visible engagement.</strong>
          <span>Copy only numbers you can actually see: views, likes, comments, shares, saves, reposts, retention, or completion rate.</span>
        </li>
        <li>
          <strong>Screenshot the proof.</strong>
          <span>Capture the engagement panel, caption/title, and post URL if visible. Save screenshots in the same folder as the CSV.</span>
        </li>
        <li>
          <strong>Leave blanks when unsure.</strong>
          <span>Do not guess. If you cannot get a clean video file, leave <code>video_file_url</code> blank and write why in <code>notes</code>, for example: “post link only; no approved download found.”</span>
        </li>
        <li>
          <strong>Send for review before upload.</strong>
          <span>Share the CSV plus the screenshot/evidence folder with Ren/Jawaun before processing.</span>
        </li>
      </ol>

      <h2>Columns To Use</h2>
      <div className="card">
        <p className="muted">Use this header row. The template download already has it.</p>
        <pre className="inline-code">{coreColumns.join(",")}</pre>
      </div>

      <h2>Example Row</h2>
      <div className="preview-table">
        <table>
          <thead>
            <tr>
              <th>video_id</th>
              <th>platform</th>
              <th>url</th>
              <th>video_file_url</th>
              <th>title</th>
              <th>creator</th>
              <th>views</th>
              <th>likes</th>
              <th>comments</th>
              <th>screenshot_url</th>
              <th>notes</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>lot-001</td>
              <td>instagram</td>
              <td>https://instagram.com/reel/example</td>
              <td>https://drive.google.com/file/d/example-video-file</td>
              <td>Why boredom matters</td>
              <td>Lectures on Tap</td>
              <td>12800</td>
              <td>740</td>
              <td>38</td>
              <td>Google Drive link to screenshot</td>
              <td>Shares/saves not visible</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>Phone Workflow</h2>
      <div className="grid">
        <article className="card">
          <h3>Instagram or TikTok</h3>
          <p>Use the share button to copy the post link. Screenshot the visible stats and caption. Then look for an approved source for the actual video file. If saves/shares are not visible, leave them blank.</p>
        </article>
        <article className="card">
          <h3>YouTube</h3>
          <p>Copy the video link. Record visible views, likes, comments, and title. If we own the video, find the original uploaded file or export. If public retention is not available, leave it blank.</p>
        </article>
        <article className="card">
          <h3>How To Save Video Files</h3>
          <p>For our videos, use the original file from Drive, the phone camera roll, Dropbox, or the editing project export. For outside accounts, only use creator-provided files, official exports, or approved downloads. Upload the file to Drive, set sharing so Ren/Jawaun can open it, then paste that Drive link into <code>video_file_url</code>.</p>
        </article>
        <article className="card">
          <h3>Folders</h3>
          <p>Create one folder per source/account. Put the CSV, screenshots, and saved video files for that source in the same folder. Name files simply, like <code>lot-001.mp4</code> and <code>lot-001-stats.png</code>.</p>
        </article>
      </div>

      <h2>Video File Checklist</h2>
      <ul className="plain-list">
        <li>Copy the public post URL into <code>url</code>. This proves which post the row means.</li>
        <li>Put the actual playable file link into <code>video_file_url</code>. This is what the model can analyze.</li>
        <li>Open the Drive link in an incognito/private browser window. If it will not play or download there, fix sharing before submitting.</li>
        <li>If there is no clean way to get the file, leave <code>video_file_url</code> blank and explain the blocker in <code>notes</code>.</li>
      </ul>

      <h2>Beginner Save Recipes</h2>
      <div className="grid">
        <article className="card">
          <h3>If We Own The Video</h3>
          <p>Find the original file in Google Drive, the phone camera roll, Dropbox, CapCut, Premiere, or wherever it was edited. Upload or move that file into the source folder. Copy the Drive file link into <code>video_file_url</code>.</p>
        </article>
        <article className="card">
          <h3>If The App Has Download</h3>
          <p>Use the normal download/save button only when the creator/account allows it. After it saves to your phone or computer, upload the file to Drive, rename it to match the row ID, and paste the Drive link into the CSV.</p>
        </article>
        <article className="card">
          <h3>If There Is No Download</h3>
          <p>Do not force it. Keep the public post link in <code>url</code>, take screenshots of the stats, leave <code>video_file_url</code> blank, and write <code>no approved video file found</code> in <code>notes</code>.</p>
        </article>
        <article className="card">
          <h3>How To Share From Drive</h3>
          <p>In Google Drive, right click or tap the three dots on the video file, choose Share, set access so Ren/Jawaun can open it, copy the link, and test that link before submitting.</p>
        </article>
      </div>

      <h2>What Not To Do</h2>
      <ul className="plain-list">
        <li>Do not guess missing numbers.</li>
        <li>Do not combine engagement into one score.</li>
        <li>Do not collect private posts or bypass access controls.</li>
        <li>Do not use sketchy downloader sites, private content, or anything that requires breaking platform rules.</li>
        <li>Do not worry about transcript, topic, thumbnail, or duration unless they are easy.</li>
      </ul>

      <p className="lede" style={{ marginTop: "1.5rem" }}>
        Ready to upload? Go to <Link href="/training-data">Training data</Link>.
      </p>
    </main>
  );
}
