import Link from "next/link";

const coreColumns = [
  "video_id",
  "platform",
  "url",
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
          Minimum useful row: video link, platform, title, creator/account, visible engagement
          numbers, and a screenshot or evidence link.
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
          <span>Copy the post URL, title/caption if visible, platform, and creator/account name.</span>
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
          <span>Do not guess. Put uncertainty in <code>notes</code>, for example: “number may be plays, not views.”</span>
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
          <p>Use the share button to copy the post link. Screenshot the visible stats and caption. If saves/shares are not visible, leave them blank.</p>
        </article>
        <article className="card">
          <h3>YouTube</h3>
          <p>Copy the video link. Record visible views, likes, comments, and title. If public retention is not available, leave it blank.</p>
        </article>
        <article className="card">
          <h3>Videos</h3>
          <p>Do not worry about downloading the video unless you are told to. A working URL is enough for now. Use official exports or approved downloads only.</p>
        </article>
        <article className="card">
          <h3>Folders</h3>
          <p>Create one folder per source/account. Put the CSV and screenshots for that source in the same folder.</p>
        </article>
      </div>

      <h2>What Not To Do</h2>
      <ul className="plain-list">
        <li>Do not guess missing numbers.</li>
        <li>Do not combine engagement into one score.</li>
        <li>Do not collect private posts or bypass access controls.</li>
        <li>Do not worry about transcript, topic, thumbnail, or duration unless they are easy.</li>
      </ul>

      <p className="lede" style={{ marginTop: "1.5rem" }}>
        Ready to upload? Go to <Link href="/training-data">Training data</Link>.
      </p>
    </main>
  );
}
